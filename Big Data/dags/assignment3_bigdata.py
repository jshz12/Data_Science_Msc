import pandas as pd
from pymongo import MongoClient 
from ucimlrepo import fetch_ucirepo 
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.models import Variable
from datetime import datetime, timedelta

def dataset_downloading(**kwargs):
    '''
	Function to download the dataset
    '''
    # Fetch dataset from UC Irvine Machine Learning Repository
    online_retail = fetch_ucirepo(id=352) 
    df = online_retail.data.original
    #print(df.shape)

    # Save dataframe to a CSV file
    file_path = '/tmp/online_retail.csv'
    df.to_csv(file_path, index=False)

    # Push the file path to XCom
    kwargs['ti'].xcom_push(key='file_path', value=file_path)

def clean_the_data(**kwargs):
    '''
	Function to clean the data
    '''
    # Pull the file path from XCom
    ti = kwargs['ti']
    file_path = ti.xcom_pull(key='file_path', task_ids='dataset_downloading')
    
    # Read the dataframe from the CSV file
    df = pd.read_csv(file_path)
    #print(df.shape)

    # Clean the data
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save the cleaned dataframe back to the CSV file
    df.to_csv(file_path, index=False)
    
    # Push the cleaned file path back to XCom
    ti.xcom_push(key='file_path', value=file_path)

def transform_the_data(**kwargs):
    '''
	Function to transform the data
    '''
    # Pull the file path from XCom
    ti = kwargs['ti']
    file_path = ti.xcom_pull(key='file_path', task_ids='clean_the_data')
    
    # Read the dataframe from the CSV file
    df = pd.read_csv(file_path)
    #print(df.shape)

    # Add a new column 'TotalPrice'
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Save the transformed dataframe back to the CSV file
    file_path = '/tmp/online_retail.csv'
    df.to_csv(file_path, index=False)
    
    # Push the transformed file path back to XCom
    ti.xcom_push(key='file_path', value=file_path)


def mongodb_loading(**kwargs):
    '''
	Function to load data into MongoDB
    '''
    # pull the file path from XCom
    ti = kwargs['ti']
    file_path = ti.xcom_pull(key='file_path', task_ids='transform_the_data')

    # read the dataframe from the csv file
    df = pd.read_csv(file_path)
    #print(df.shape)
    retail_data = df.to_dict(orient='records')
    #print(retail_data)
    client = MongoClient('mongodb://mymongo:27017')

    # specify the database to use
    db = client['transformed_data']

    # specify the collection to use
    collection = db['online_retail_dataset']

    # insert data into the collection
    result = collection.insert_many(retail_data)
    #print(result)

# fetch the emails that will recieve the notification
emails = Variable.get("emails")
email_list = [value.strip() for value in emails.split(',')]

# defaults
default_args = {
    'owner': 'David',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 21),
    'email': email_list,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG('Assignment3', default_args=default_args, description="Big Data Assignment 3 by David Íñiguez and Jaume Sánchez", schedule_interval=timedelta(days=1))

# Task defining using operators
download_task = PythonOperator(
    task_id='dataset_downloading',
    python_callable=dataset_downloading,
    dag=dag)

cleaning_task = PythonOperator(
    task_id='clean_the_data',
    python_callable=clean_the_data,
    dag=dag)

transformation_task = PythonOperator(
    task_id='transform_the_data',
    python_callable=transform_the_data,
    dag=dag)

nosql_task = PythonOperator(
    task_id='mongodb_loading',
    python_callable=mongodb_loading,
    dag=dag)

# Extra Exercise

summary_email = EmailOperator(
    task_id='send_summary_email',
    to=email_list,
    subject='DAG {{ dag.dag_id }}: Summary',
    html_content="""
    <h3>Summary for DAG {{ dag.dag_id }}</h3>
    <p>The DAG {{ dag.dag_id }} has completed its run.</p>
    <p>Run details:</p>
    <ul>
        <li>Execution Date: {{ ds }}</li>
        <li>Start Date: {{ execution_date }}</li>
        <li>End Date: {{ next_execution_date }}</li>
    </ul>
    """,
    dag=dag
)

#DAG
download_task >> cleaning_task >> transformation_task >> nosql_task >> summary_email
