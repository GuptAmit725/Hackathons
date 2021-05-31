from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


def tfidf(data_col, data_tr_col):
    vectorizer = TfidfVectorizer(max_features = 50)
    vectorizer.fit(data_tr_col)
    return vectorizer.transform(data_col).todense()

def predict(classifier,X_tr,stats,occupation,channel,is_active,age,vintage_month):
  inp0 = pd.DataFrame([occupation],columns = ['Occupation'])
  inp1 = pd.DataFrame([channel],columns = ['Channel_Code'])
  inp2 = pd.DataFrame([is_active],columns = ['Is_Active'])


  inp00 = tfidf(inp0['Occupation'], X_tr['Occupation'])
  inp10 = tfidf(inp1['Channel_Code'], X_tr['Channel_Code'])
  inp20 = tfidf(inp2['Is_Active'], X_tr['Is_Active'])

  u_inp =  np.column_stack((
                            inp00,
                            inp10,
                            inp20
                          ))
  u_inp_c = np.append(np.array(u_inp),[[age],[vintage_month]])
  for i,mean in enumerate(stats[0]):
    u_inp_c[i] -= mean /  np.sqrt(stats[1][i])

  pred = classifier.predict(u_inp_c.reshape(-1,1).T)[0]

  return pred