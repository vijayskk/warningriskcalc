import joblib

# we have to save
# 1. bow_transformer
# 2. tfidf_transformer
# 3. model

model = joblib.load('saves/model.joblib')
bow_transformer = joblib.load('saves/bow_transformer.joblib')
tfidf_transformer = joblib.load('saves/tfidf_transformer.joblib')



def findRisk(messages):
    bown = bow_transformer.transform(messages)
    tfidfn = tfidf_transformer.transform(bown)
    return model.predict(tfidfn)

# result = findRisk(["Connection lost"])
# print(result)
risk = findRisk([input("\nEnter the error: ")])
print(f"\nThe warning have {risk[0].upper()} risk.\n")