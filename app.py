from flask import Flask, render_template, request
import pickle
import numpy as np

Medicine = pickle.load(open('medicine.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
med = pickle.load(open('med.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_medicine')
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = med[med['drugName'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('drugName')["drugName"].values))
        item.extend(list(temp_df.drop_duplicates('drugName')['condition'].values))
        item.extend(list(temp_df.drop_duplicates('drugName')['review'].values))
        item.extend(list(temp_df.drop_duplicates('drugName')['rating'].values))

        data.append(item)

    print(data)

    return render_template('recommend.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)

