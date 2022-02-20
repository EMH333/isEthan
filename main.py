import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import data
from data import get_combinations


def get_data():
    col = ['isEthan']
    col.extend(get_combinations())
    csv = pd.read_csv('data/grams.data', names=col, sep=',')
    return csv


def into_form(test_text):
    wordmap = data.analyze(test_text)
    inp = list()
    for i in data.combos:
        try:
            inp.append(wordmap[i])
        except KeyError:
            inp.append(0)
    return inp


if __name__ == '__main__':
    scaler = StandardScaler()
    df = get_data()
    X = df.drop('isEthan', axis=1)
    y = df[['isEthan']]
    actual_y = list()
    for xy in y.values:
        actual_y.append(xy[0])
    X_train, X_test, y_train, y_test = train_test_split(X, actual_y, test_size=0.2)
    # make sure we are scaled correctly
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # rf_clf = RandomForestClassifier(criterion='entropy')
    rf_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                           hidden_layer_sizes=(100, 10), random_state=1)
    rf_clf.fit(X_train, y_train)
    y_predict = rf_clf.predict(X_test)
    print("Accuracy score of " + str(accuracy_score(y_test, y_predict)))

    final_test_text_true = "So why do I care about student government? I care because I help approve a 27 million dollar budget that directly funds things students need to survive. I care because I have a better chance to promote inclusion and justice at the local level. I care because the more educated and better prepared this generation is, the more likely we will be able to turn the tide of climate change. But I don't care a lot. There are guard rails in place to make sure the decisions we make are somewhat sane. It isn't worth my time to care a lot, because if I do then it becomes part of my identity. Politics comes with a lot of drama and it we want to make lasting change, we have to filter all that out. I don't protect the president or vice-president, I don't protect ASOSU, because I can see that my life has value beyond those opinions."
    final_test_text_false = "It was that kind of summer, Ms. Jones said, when the weather changed so suddenly that she developed a novel hack for managing the heat. “Every time we drive past a fire hydrant,” she said, “I try to open the windows so the kids get wet.”On this July afternoon, Ms. Jones, a Pentecostal minister, was driving along a computer-generated route — windows closed — as part of a [“citizen scientist” project](https://www.capastrategies.com/capa-heat-watch) to map the way heat pooled in certain sections of New York. Nine cars drove simultaneously along different routes in Upper Manhattan and the South Bronx, each bearing a sensor that measured air temperature and humidity once per second.Data from the project, which will not be available until the fall, will add detail to what climate and health researchers have known for years: that heat in New York is distributed unequally, and that its distribution follows other patterns of inequality, including race, income, air quality and access to air-conditioning."
    final_data = scaler.transform([into_form(final_test_text_true), into_form(final_test_text_false)])

    answers = rf_clf.predict(final_data)
    print(answers)
