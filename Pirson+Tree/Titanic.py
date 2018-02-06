import pandas
import numpy
import math
import re

def countgender(data):
    male = round((data['Sex'].value_counts()['male']),2)
    female = round((data['Sex'].value_counts()['female']),2)
    fo = open("male_female.txt", "w")
    fo.write(str(male) + " " + str(female));
    fo.close()

def countalive(data):
    part = round((data['Survived'].value_counts(bin(0))[1])*100,2)
    fo = open("alive.txt", "w")
    fo.write(str(part));
    fo.close()

def countPclass(data):
    part = round((data['Pclass'].value_counts(bin(0))[1])*100,2)
    fo = open("1class.txt", "w")
    fo.write(str(part));
    fo.close()

def passengersage(data):
    av = round((data['Age'].mean()),2)
    median = round((data['Age'].median()),2)
    fo = open("av_med.txt", "w")
    fo.write(str(av) + " " + str(median));
    fo.close()

def correlation(data):
    SibSp = numpy.asarray(data['SibSp'].tolist())
    Parents_child = numpy.asarray(data['Parch'].tolist())

    S_m = SibSp.mean()
    P_m = Parents_child.mean()
    temp = SibSp*Parents_child
    upperpart = (SibSp*Parents_child).mean()-S_m*P_m
    downpart = (math.sqrt((SibSp**2).mean()-S_m**2))*(math.sqrt((Parents_child**2).mean()-P_m**2))

    Pirson = round(upperpart/downpart,2)

    fo = open("Pirson.txt", "w")
    fo.write(str(Pirson));
    fo.close()

def extract_first_name(name):
    """
    Функция извлечения first name from name
    :param name: name
    :return: first name
    """
    # первое слово в скобках
    m = re.search(".*\\((.*)\\).*", name)
    if m is not None:
        return m.group(1).split(" ")[0]
    # первое слово после Mrs. or Miss. or else
    m1 = re.search(".*\\. ([A-Za-z]*)", name)
    return m1.group(1)

def first_name(data):
    fn = data[data['Sex'] == 'female']['Name']
    name = fn.map(lambda full_name: extract_first_name(full_name)).value_counts().idxmax()
    fo = open("Name.txt", "w")
    fo.write(str(name));
    fo.close()


data = pandas.read_csv('titanic.csv', index_col='PassengerId')


countgender(data)

countalive(data)

countPclass(data)

passengersage(data)

correlation(data)

first_name(data)




