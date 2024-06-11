import pickle
import csv
from sklearn.feature_extraction.text import TfidfVectorizer

fitted_vectorizer = None

def fit_vectorizer_with_inputs(input_strings):
    global fitted_vectorizer
    fitted_vectorizer = TfidfVectorizer()
    fitted_vectorizer.fit(input_strings)

def take_input():
    user_input = input("Enter a string to convert into vector using TF-IDF: ")
    return user_input

def string_to_vector(string):
    global fitted_vectorizer
    vector = fitted_vectorizer.transform([string])
    return vector

def store_csv(data):
    with open('data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

def main():
    # Load existing strings from CSV
    existing_strings = []
    with open('data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            existing_strings.append(row[0])

    # Fit the vectorizer with the existing strings
    fit_vectorizer_with_inputs(existing_strings)

    user_input = take_input()
    vector = string_to_vector(user_input)

    with open('vector.pkl', 'wb') as file:
        pickle.dump(vector, file)
    print("Vector has been stored in 'vector.pkl' file.")

    # Convert vector to array and then list for CSV storage
    vector_list = vector.toarray().tolist()
    store_csv(vector_list)
    print("Vector has been stored in 'data.csv' file as a single column.")

if __name__ == "__main__":
    main()
