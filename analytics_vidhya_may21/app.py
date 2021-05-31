from flask import Flask, render_template, request
import pickle
import prediction


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST']) # To render Homepage
def home_page():
    return render_template('index.html')

@app.route('/math', methods=['POST'])  # This will be called from UI
def math_operation():

    if (request.method=='POST'):
        occupation=request.form['occupation']
        Channel = request.form['Channel']
        Is_active = request.form['Is_active']
        age=int(request.form['age'])
        vintage_month = int(request.form['vintage_month'])


        X_tr = pickle.load(open('X_tr.pkl', 'rb'))
        classifier = pickle.load(open('dtree_best_model.pkl', 'rb'))
        stats = pickle.load(open('mean_var_age_and_vintage_list.pkl', 'rb'))

        pred = prediction.predict(classifier, X_tr,stats,occupation,Channel,Is_active,age,vintage_month)
        return render_template('results.html',result=pred)


if __name__ == '__main__':
    app.run(debug=True)
