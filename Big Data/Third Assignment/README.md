### DAVID ÍÑIGUEZ & JAUME SÁNCHEZ
#### README with instructions on how to set up and run this Airflow project

### 1. Directory Structure
First, create a directory (we named it as airflow) with the following folders: 
1)dags:   It will contain the dags, specified in the:  assignment3_bigdata.py
2)logs: It will storage the logs from Airflow showing successful execution of the DAG.
3)plugins
4) docker-compose.yml
5) env:  file containing the following: AIRFLOW_UID = 5000


### 2. Email task
To ensure your DAG runs smoothly and sends email notifications with process summaries, you will need to change some configure SMTP settings, editing the docker-compose.yml file, specifically adjusting the parameters for HOST, USER, PASSWORD, and MAIL_FROM to enable proper email functionality. Note that in order to make it run, we changed this variables to our email but now it does not appear, to make it reproducible.

```
    AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
    AIRFLOW__SMTP__SMTP_STARTTLS: 'true'
    AIRFLOW__SMTP__SMTP_SSL: 'false'
    AIRFLOW__SMTP__SMTP_USER: youremail@example.com
    AIRFLOW__SMTP__SMTP_PASSWORD: yourpassword
    AIRFLOW__SMTP__SMTP_PORT: '587'
    AIRFLOW__SMTP__SMTP_MAIL_FROM: youremail@example.com
```

### 3. docker compose up
When you have all the aforementioned settings ready, just run docker compose up (within the airflow directory being set)

### 4. UI Browser Access

You can now access the UI in a browser writing localhost:8080 and using airflow as both user and password.

### 5. Email Notifications
Finally, once you have log in you should go to Admin section and then to Variables
There, you should create a new variable named 'emails' with the email notificacion addresses you want the notifications to be sent
Note that if you want more than one email adress you should separate them by commas.
Do not forget to Save the 'email' variable.


### 6. ETL Ready 
It is all ready to work, enjoy!