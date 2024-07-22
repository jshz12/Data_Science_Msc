from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures
import re
import nltk
from nltk.corpus import stopwords

# List of Prepositions
prepositions = ["about", "above", "across", "after", "against", "along", "among", "around", "at",
                    "before", "behind", "below", "beneath", "beside", "between", "beyond", "by",
                    "despite", "down", "during", "except", "for", "from", "in", "inside", "into",
                    "like", "near", "of", "off", "on", "out", "outside", "over", "past", "since",
                    "through", "throughout", "till", "to", "toward", "under", "underneath", "until", 
                    "up", "upon", "with", "within", "without"]

# List of Prefixes and Suffixes
prefixes = ["Proto", "Neo", "Hyper", "Ultra", "Mega", "Nano", "Micro", "Mini", "Pre", "Post", "Mid", "Bi", "Tri", "Quadri", "Penta", "Hexa", "Deca", "Century","Millennium", "Decade","Inter", "Trans", "Geo", "South", "Supra", "Trans", "Inter", "Supra", "Sub","Geo", "Astro", "Hydro", "Thermo", "Cryo", "Atmo", "Bio", "Chem", "Eco", "Seismo", "Volcano", "Ltd.", "Corp.", "Co.", "Assoc.", "Inst.", "Mrs.", "Mr.", "Prof.", "Capt.", "Sgt.", "Dr.", "Pre", "Post", "Mid", "Ante", "Fore", "After"]

suffixes = ["ware", "bot", "drone", "craft", "ship", "car", "bike", "phone", "gram", "scope","meter", "fest", "thon", "con", "gala", "fair", "meet", "show", "summit", "convention", "ville", "ford", "stan", "shire", "burg", "grad", "port", "land", "stan", "nia", "desh", "ville", "burgh", "sia", "ada", "ana", "ish", "ese", "quake", "storm", "wave", "wind", "rain", "snow", "fall", "rise", "set", "light", "clipse", "Inc", "Ltd", "Corp", "Co", "Inst",     "son", "man", "berg", "stein", "sen", "smith", "day", "night", "week", "month", "year", "decade", "century", "millennium"]


# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.stop_words = set(stopwords.words('english'))

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)

        # Get word name from ID.
        if isinstance(x, str):
            x_name = x
        else:
            x_name = self.dataset.x_dict.get_label_name(x)

        word = str(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

    
        # CONTRIBUTION TO ORIGINAL CLASS
        # Adding additional features

        # First Uppercase Check
        if word[0].isupper():
            first_cap_feat_name = "capitalized::{}".format(y_name)
            first_cap_feat_id = self.add_feature(first_cap_feat_name)
            if first_cap_feat_id != -1:
                features.append(first_cap_feat_id)

        # All Uppercase Check
        if word.isupper():
            all_cap_feat_name = "all_capitalized::{}".format(y_name)
            all_cap_feat_id = self.add_feature(all_cap_feat_name)
            if all_cap_feat_id != -1:
                features.append(all_cap_feat_id)

        # All Lowercase Check
        if word.islower():
            lower_feat_name = "lower::{}".format(y_name)
            lower_feat_id = self.add_feature(lower_feat_name)
            if lower_feat_id != -1:
                features.append(lower_feat_id)

        # Digit Check
        if any(char.isdigit() for char in word):
            digit_feat_name = "contains_digit::{}".format(y_name)
            digit_feat_id = self.add_feature(digit_feat_name)
            if digit_feat_id != -1:
                features.append(digit_feat_id)

        # Alphanumeric Check
        if word.isalnum():
            alphanum_feat_name = "alphanum::{}".format(y_name)
            alphanum_feat_id = self.add_feature(alphanum_feat_name)
            if alphanum_feat_id != -1:
                features.append(alphanum_feat_id)

        # Punctuation Check
        if word in '.,;:!?()[]{}':
            punct_feat_name = "punctuation::{}".format(y_name)
            punct_feat_id = self.add_feature(punct_feat_name)
            if punct_feat_id != -1:
                features.append(punct_feat_id)

        # Hyphen Check
        if '-' in word:
            hyphen_feat_name = "hyphen::{}".format(y_name)
            hyphen_feat_id = self.add_feature(hyphen_feat_name)
            if hyphen_feat_id != -1:
                features.append(hyphen_feat_id)

        # Stopword Check
        if word.lower() in self.stop_words:
            stopword_feat_name = "stopword::{}".format(y_name)
            stopword_feat_id = self.add_feature(stopword_feat_name)
            if stopword_feat_id != -1:
                features.append(stopword_feat_id)
        
        # Preposition Check
        if word in prepositions:
            preposition_feat_name = "preposition::{}".format(y_name)
            preposition_feat_id = self.add_feature(preposition_feat_name)
            if preposition_feat_id != -1:
                features.append(preposition_feat_id)

        # Prefixes and Suffixes Checks

        for prefix in prefixes:
            if word.startswith(prefix):
                prefix_feat_name = "prefix:{}::{}".format(prefix, y_name)
                prefix_feat_id = self.add_feature(prefix_feat_name)
                if prefix_feat_id != -1:
                    features.append(prefix_feat_id)
        
        for suffix in suffixes:
            if word.endswith(suffix):
                suffix_feat_name = "suffix:{}::{}".format(suffix, y_name)
                suffix_feat_id = self.add_feature(suffix_feat_name)
                if suffix_feat_id != -1:
                    features.append(suffix_feat_id)

        return features
