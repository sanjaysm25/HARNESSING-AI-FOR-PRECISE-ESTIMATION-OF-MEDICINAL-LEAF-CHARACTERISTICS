import os
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
import joblib
from PIL import Image
import io 
import tensorflow as tf

# Load the VGG16 model with the specified weights file
base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
model_vgg = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv2').output)

# Load the Random Forest classifier
rf_classifier = joblib.load('model_random_forest_dataset2.pkl')

# Load the trained VGG16 model for classification
model = tf.keras.models.load_model('model_vgg_dataset2.h5')

# Get the class names for the VGG16 model
class_names = sorted(os.listdir('dataset2'))  # replace with your actual directory

data = {
    "Alpinia Galanga (Rasna)": {
        "medicinal_properties": ["Anti-inflammatory", "Antioxidant", "Antimicrobial"],
        "geo_location": ["India ", "Southeast Asia"],
        "disease_curable": ["Arthritis", "Indigestion"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": False,
            "recommended_age": "Not recommended for children under 15",
            "pregnancy_consideration": "Avoid use during pregnancy due to potential risks"
        },
        "medicinal_values": [
            "Used for digestive issues and respiratory ailments",
            "Has anti-inflammatory properties",
            "Used in traditional medicine for pain relief"
        ],
        "side_effects": [
            "Potential gastrointestinal discomfort",
            "Allergic reactions possible",
            "Not advised during pregnancy due to potential complications for the fetus"
        ],
        "note": "Consult a healthcare professional before use, especially during pregnancy or for children"
    },

     "Amaranthus Viridis (Arive-Dantu)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Antibacterial"],
        "geo_location": ["Worldwide"],
        "disease_curable": ["Asthma", "Diabetes", "High Blood Pressure"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Not recommended for pregnant women",
            "pregnancy_consideration": "Not recommended for pregnant women due to possible complications"
        },
        "medicinal_values": [
            "Rich in vitamins and minerals",
            "Supports digestion and immune health",
            "May aid in managing diabetes and high cholesterol"
        ],
        "side_effects": [
            "Possible allergic reactions",
            "Can cause digestive discomfort in some individuals",
            "Avoid during pregnancy due to potential adverse effects on the fetus"
        ],
        "note": "Consult a healthcare professional before use, especially during pregnancy or for children"
    },

    "Artocarpus Heterophyllus (Jackfruit)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Digestive Disorders", "Skin Diseases"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for all ages, including children and adults",
            "pregnancy_consideration": "Limited data on safety during pregnancy; consult a doctor"
        },
        "medicinal_values": [
            "Rich in nutrients like vitamin C and fiber",
            "Supports digestion and may aid in regulating blood sugar levels"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "High in sugar and can cause gastrointestinal discomfort if consumed in excess",
            "Limited evidence on safety during pregnancy, so caution advised"
        ],
        "note": "Moderation is key, especially during pregnancy. Consult a healthcare professional for personalized advice"
    },

   "Azadirachta Indica (Neem)": {
        "medicinal_properties": ["Antibacterial", "Antifungal", "Antiviral"],
        "geo_location": ["India"],
        "disease_curable": ["Skin Diseases", "Malaria", "Diabetes"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited safety data for pregnant women and children under 2"
        },
        "medicinal_values": [
            "Antimicrobial and anti-inflammatory properties",
            "Used in traditional medicine for skin conditions, dental health, and digestive issues"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Ingestion of large amounts may cause stomach upset",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use caution, especially during pregnancy and for young children. Consult a doctor before use"
    },

   "Basella Alba (Basale)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Anticancer"],
        "geo_location": ["India"],
        "disease_curable": ["Constipation", "Anemia"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in vitamins and minerals",
            "Supports eye health and aids digestion",
            "Traditionally used for its anti-inflammatory properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may lead to digestive discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

    "Brassica Juncea (Indian Mustard)": {
        "medicinal_properties": ["Antibacterial", "Antifungal", "Anticancer"],
        "geo_location": ["India"],
        "disease_curable": ["Cough", "Asthma", "Bronchitis"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in vitamins and minerals, especially vitamin K and folate",
            "Supports heart health and aids digestion",
            "Contains antioxidants with potential anti-cancer properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "High consumption may lead to digestive discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

        "Carissa Carandas (Karanda)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Antidiabetic"],
        "geo_location": ["India"],
        "disease_curable": ["Diabetes", "Wounds", "Ulcers"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in vitamin C and antioxidants",
            "Supports digestive health and may boost immunity",
            "Used in traditional medicine for treating diarrhea and skin ailments"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may lead to digestive discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },
    "Citrus Limon (Lemon)": {
        "medicinal_properties": ["Antioxidant", "Antibacterial", "Antiviral"],
        "geo_location": ["Worldwide"],
        "disease_curable": ["Scurvy", "Indigestion", "Skin Care"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in vitamin C and antioxidants",
            "Supports immune health and aids digestion",
            "Used in traditional medicine for its antibacterial properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "High consumption may erode tooth enamel",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

    "Ficus Auriculata (Roxburgh Fig)": {
        "medicinal_properties": ["Antioxidant", "Antidiabetic", "Antimicrobial"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Diabetes", "Skin Diseases", "Wounds"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in fiber and antioxidants",
            "Supports digestive health and may help regulate blood sugar levels",
            "Used in traditional medicine for treating skin conditions"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may cause digestive discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },
    "Ficus Religiosa (Peepal Tree)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Antidiabetic"],
        "geo_location": ["India"],
        "disease_curable": ["Asthma", "Jaundice", "Diabetes"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in antioxidants and vitamins",
            "Supports digestive health and may aid in managing diabetes",
            "Used in traditional medicine for treating wounds and skin ailments"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may lead to digestive discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },
   "Hibiscus Rosa-sinensis": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Antibacterial"],
        "geo_location": ["Tropical Regions"],
        "disease_curable": ["Hair Loss", "Hypertension", "Cough"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in antioxidants and vitamin C",
            "Supports cardiovascular health and may help lower blood pressure",
            "Used in traditional medicine for its diuretic properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "May interact with certain medications",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

    "Jasminum (Jasmine)": {
        "medicinal_properties": ["Antidepressant", "Antiseptic", "Antispasmodic"],
        "geo_location": ["Tropical Regions"],
        "disease_curable": ["Anxiety", "Skin Diseases", "Menstrual Disorders"]
    },
    "Jasminum (Jasmine)": {
        "medicinal_properties": ["Antidepressant", "Antiseptic", "Antispasmodic"],
        "geo_location": ["Tropical Regions"],
        "disease_curable": ["Anxiety", "Skin Diseases", "Menstrual Disorders"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Used in aromatherapy for stress relief and relaxation",
            "Some traditional uses include improving digestion and skin health"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Use caution with essential oil applications, especially in young children",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },
    "Mangifera Indica (Mango)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Anticancer"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Indigestion", "Heat Stroke", "Anemia"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in vitamins A, C, and E",
            "Supports immune health and aids digestion",
            "Contains antioxidants with potential anti-inflammatory properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may cause gastrointestinal discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

    "Mentha (Mint)": {
        "medicinal_properties": ["Antimicrobial", "Antispasmodic", "Digestive Aid"],
        "geo_location": ["Worldwide"],
        "disease_curable": ["Indigestion", "Nausea", "Headache"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Aids digestion and relieves nausea",
            "Soothes headaches and nasal congestion",
            "Used in traditional medicine for its antimicrobial properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may cause heartburn",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },
    "Moringa Oleifera (Drumstick)": {
        "medicinal_properties": ["Antioxidant", "Anti-inflammatory", "Antidiabetic"],
        "geo_location": ["India", "Africa"],
        "disease_curable": ["Diabetes", "Anemia", "Malnutrition"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in vitamins, minerals, and antioxidants",
            "Supports immune health and aids digestion",
            "Traditionally used for its anti-inflammatory properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "High doses may cause digestive discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

    "Muntingia Calabura (Jamaica Cherry-Gasagase)": {
        "medicinal_properties": ["Antioxidant", "Antimicrobial", "Anti-inflammatory"],
        "geo_location": ["Tropical Regions"],
        "disease_curable": ["Fever", "Hypertension", "Diabetes"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in vitamin C and antioxidants",
            "Supports immune health and aids digestion",
            "Traditionally used for its anti-inflammatory properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "High doses may cause digestive discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

   "Murraya Koenigii (Curry)": {
        "medicinal_properties": ["Antioxidant", "Antimicrobial", "Anti-inflammatory"],
        "geo_location": ["India"],
        "disease_curable": ["Diabetes", "Diarrhea", "Nausea"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in vitamins and minerals",
            "Supports digestion and may help manage diabetes",
            "Traditionally used for its anti-inflammatory properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "High doses may cause gastrointestinal discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

   "Nerium Oleander (Oleander)": {
        "medicinal_properties": ["Cardiotonic", "Anticancer", "Antimicrobial"],
        "geo_location": ["Mediterranean Region", "Asia"],
        "disease_curable": ["Heart Diseases", "Cancer"],
        "age_limit": {
            "safe_for_adults": False,
            "safe_for_children": False,
            "recommended_age": "Not safe for children or adults",
            "pregnancy_consideration": "Highly toxic, ingestion can be fatal"
        },
        "medicinal_values": [
            "No safe medicinal uses",
            "Contains cardiac glycosides, used in traditional medicine with extreme caution"
        ],
        "side_effects": [
            "Severe toxicity, including cardiac arrhythmias and death",
            "Skin contact may cause irritation or allergic reactions",
            "Not safe during pregnancy, can harm the fetus"
        ],
        "note": "Avoid use entirely, especially for pregnant women and children. Immediate medical attention required if ingested"
    },

    "Nyctanthes Arbor-tristis (Parijata)": {
        "medicinal_properties": ["Antipyretic", "Antiarthritic", "Antioxidant"],
        "geo_location": ["India"],
        "disease_curable": ["Fever", "Arthritis", "Skin Diseases"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Used in traditional medicine for treating fever and respiratory issues",
            "Has anti-inflammatory and analgesic properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may cause gastrointestinal discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

   "Ocimum Tenuiflorum (Tulsi)": {
        "medicinal_properties": ["Antioxidant", "Antimicrobial", "Anti-inflammatory"],
        "geo_location": ["India"],
        "disease_curable": ["Cough", "Cold", "Insect Bites"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Supports respiratory health and boosts immunity",
            "Used in traditional medicine for its anti-inflammatory and antimicrobial properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may lower blood sugar levels",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

    "Piper Betle (Betel)": {
        "medicinal_properties": ["Antibacterial", "Antifungal", "Antioxidant"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Oral Health", "Digestive Disorders"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Used in traditional medicine for oral health and digestion",
            "Contains antioxidants and antimicrobial properties"
        ],
        "side_effects": [
            "May cause oral cancer when combined with tobacco or areca nut",
            "Excessive chewing may lead to addiction and stained teeth",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use with caution, especially during pregnancy and for young children. Avoid combining with tobacco or areca nut. Consult a doctor before use"
    },

     "Plectranthus Amboinicus (Mexican Mint)": {
        "medicinal_properties": ["Antibacterial", "Antifungal", "Antioxidant"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Respiratory Disorders", "Digestive Disorders"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Used in traditional medicine for respiratory issues and digestive problems",
            "Has antimicrobial and anti-inflammatory properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may cause gastrointestinal discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

   "Pongamia Pinnata (Indian Beech)": {
        "medicinal_properties": ["Antibacterial", "Antifungal", "Antioxidant"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Skin Diseases", "Wounds", "Rheumatism"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Used in traditional medicine for skin ailments and wound healing",
            "Has antimicrobial and anti-inflammatory properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may cause digestive discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },
   "Psidium Guajava (Guava)": {
        "medicinal_properties": ["Antioxidant", "Antimicrobial", "Anti-inflammatory"],
        "geo_location": ["Tropical Regions"],
        "disease_curable": ["Diarrhea", "Dysentery", "Skin Disorders"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in vitamin C and antioxidants",
            "Supports immune health and aids digestion",
            "May help regulate blood sugar levels"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may cause gastrointestinal discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

    "Punica Granatum (Pomegranate)": {
        "medicinal_properties": ["Antioxidant", "Anticancer", "Antimicrobial"],
        "geo_location": ["Middle East", "India"],
        "disease_curable": ["Heart Diseases", "Diabetes", "High Blood Pressure"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in antioxidants and vitamin C",
            "Supports heart health and aids digestion",
            "May help reduce inflammation and improve memory"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may cause digestive discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

    "Santalum Album (Sandalwood)": {
        "medicinal_properties": ["Antiseptic", "Anti-inflammatory", "Astringent"],
        "geo_location": ["India", "Australia"],
        "disease_curable": ["Skin Diseases", "Urinary Tract Infections"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Used in aromatherapy for relaxation and mental clarity",
            "Has anti-inflammatory and antiseptic properties",
            "Traditional remedy for skin conditions like acne and eczema"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Skin irritation may develop with direct application",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

   "Syzygium Cumini (Jamun)": {
        "medicinal_properties": ["Antidiabetic", "Antioxidant", "Antimicrobial"],
        "geo_location": ["India"],
        "disease_curable": ["Diabetes", "Digestive Disorders", "Skin Diseases"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in antioxidants and vitamins",
            "Supports blood sugar control and aids digestion",
            "Traditionally used for its antimicrobial properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may cause digestive discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

    "Syzygium Jambos (Rose Apple)": {
        "medicinal_properties": ["Antioxidant", "Antimicrobial", "Anticancer"],
        "geo_location": ["Southeast Asia"],
        "disease_curable": ["Diabetes", "Digestive Disorders", "Cancer"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Rich in vitamins and antioxidants",
            "Supports digestive health and boosts immunity"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may cause gastrointestinal discomfort"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },

    "Tabernaemontana Divaricata (Crape Jasmine)": {
        "medicinal_properties": ["Antipyretic", "Antispasmodic", "Anti-inflammatory"],
        "geo_location": ["India", "Southeast Asia"],
        "disease_curable": ["Fever", "Muscle Pain", "Inflammation"],
        "age_limit": {
            "safe_for_adults": True,
            "safe_for_children": True,
            "recommended_age": "Generally safe for adults and children over 2",
            "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
        },
        "medicinal_values": [
            "Used in traditional medicine for reducing fever and muscle pain",
            "Has anti-inflammatory and antispasmodic properties"
        ],
        "side_effects": [
            "Rarely, allergic reactions may occur",
            "Excessive consumption may cause gastrointestinal discomfort",
            "Limited safety data during pregnancy; consult a healthcare professional"
        ],
        "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use"
    },
    "Trigonella Foenum-graecum (Fenugreek)": {
    "medicinal_properties": ["Antidiabetic", "Antioxidant", "Anti-inflammatory"],
    "geo_location": ["India", "Mediterranean Region"],
    "disease_curable": ["Diabetes", "Digestive Disorders", "Skin Inflammation"],
    "age_limit": {
        "safe_for_adults": True,
        "safe_for_children": True,
        "recommended_age": "Generally safe for adults and children over 2",
        "pregnancy_consideration": "Limited data for pregnant women and children under 2; consult a doctor"
    },
    "medicinal_values": [
        "Supports lactation in nursing mothers.",
        "May help regulate blood sugar levels and aid digestion.",
        "Used in traditional medicine for its anti-inflammatory properties."
    ],
    "side_effects": [
        "Rarely, allergic reactions may occur.",
        "Excessive consumption may cause gastrointestinal discomfort.",
        "Limited safety data during pregnancy; consult a healthcare professional."
    ],

    "note": "Use in moderation, especially during pregnancy and for young children. Consult a doctor before use."
},
}

# Function to extract features using VGG16
def extract_features(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))  # Resize the image to match VGG input
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    with tf.name_scope("extract_features"):
        features = model_vgg.predict(x)
    
    return features.flatten()

# Function to predict using Random Forest
def predict_rf(image_bytes):
    new_image_features = extract_features(image_bytes)
    prediction_features = new_image_features.reshape(1, -1)
    prediction = rf_classifier.predict(prediction_features)
    return prediction

# Function to load and prepare the image for VGG16 model
def load_and_prep_image(image, img_shape=224):
    """
    Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape,, color_channels)
    """
    # Decode it into a tensor
    img = tf.image.decode_image(image)

    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])

    # Rescale the image (get all values between 0 and 1)
    img = img/255.
    return img

# Streamlit app
st.title("🌿 HARNESSING AI FOR PRECISE ESTIMATION OF MEDICAL LEAF CHARACTERISTICS 🌿")

# Add custom CSS for dark mode


uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    dataset_dir = 'C:/Users/hp/Desktop/mAJOR PROJECT/Aryvedic/major project codes/major project final codes/dataset2'
  
    # Check if the uploaded image is in dataset2 directory
    in_dataset2 = False
    for root, dirs, files in os.walk(dataset_dir):
        if uploaded_file.name in files:
            in_dataset2 = True
            break
    
    if in_dataset2:
        # Predict with Random Forest
        prediction_rf = predict_rf(image_bytes)
        pred_class_rf = prediction_rf[0]

        # Predict with VGG16
        image_vgg = load_and_prep_image(image_bytes)
        image_vgg = tf.expand_dims(image_vgg, axis=0)
        pred_vgg = model.predict(image_vgg)
        pred_class_vgg = class_names[np.argmax(pred_vgg)]

        if pred_class_vgg == pred_class_rf:
            pred_class = pred_class_vgg
            st.write("Based on Majority Voting:")
            st.write(f"Prediction: {pred_class}")

            if pred_class in data:
                st.markdown(f"**Medicinal Properties:** {', '.join(data[pred_class]['medicinal_properties'])}")
                st.markdown(f"**Geo Location:** {', '.join(data[pred_class]['geo_location'])}")
                st.markdown(f"**Disease Curable:** {', '.join(data[pred_class]['disease_curable'])}")
                st.markdown(f"**Age Limit:** {data[pred_class]['age_limit']}")
                st.markdown("**Medicinal Values:**")
                for value in data[pred_class]['medicinal_values']:
                    st.markdown(f"- {value}")
                st.markdown("**Side Effects:**")
                for effect in data[pred_class]['side_effects']:
                    st.markdown(f"- {effect}")
                st.markdown(f"**Note:** {data[pred_class]['note']}")
            else:
                st.write("No additional data available")
        else:
            st.write("Predictions from Individual Models:")
            if pred_class_rf in class_names:
                pred_class = pred_class_rf
                st.markdown(f"Prediction  : **{pred_class}**")
                if pred_class in data:
                    st.markdown(f"**Medicinal Properties:** {', '.join(data[pred_class]['medicinal_properties'])}")
                    st.markdown(f"**Geo Location:** {', '.join(data[pred_class]['geo_location'])}")
                    st.markdown(f"**Disease Curable:** {', '.join(data[pred_class]['disease_curable'])}")
                    st.markdown(f"**Age Limit:** {data[pred_class]['age_limit']}")
                    st.markdown("**Medicinal Values:**")
                    for value in data[pred_class]['medicinal_values']:
                        st.markdown(f"- {value}")
                    st.markdown("**Side Effects:**")
                    for effect in data[pred_class]['side_effects']:
                        st.markdown(f"- {effect}")
                    st.markdown(f"**Note:** {data[pred_class]['note']}")
                else:
                    st.write("No additional data available")
            else:
                st.write(" Image")
    else:
        st.write(" ❌ THE UPLOADED IMAGE DATA IS NOT FOUND  ❌")
