#!/usr/bin/env python
# coding: utf-8

# ## BIBLIOTECAS

# In[1]:


import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report


# ## Tratamentos dos dados
# carregando dados

# In[3]:


df=pd.read_excel(R'C:\Users\Anderson Salata\Documents\Testes Analista Dados Sr\Case_Formatura.xlsx')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.sample(3).T
#conferindo a importação dos dados.


# In[17]:


df.info()
#verificando os tipos de variáveis existentes no banco de dados


# ####

# In[7]:


df.columns.values


# In[8]:


df.corr().T
#conhecendo as correlações das variáveis do tipo inteiro.


# In[9]:


msno.matrix(df)
#usando a matrix para analisar dados em brancos e possíveis erros.


# In[10]:


msno.bar(df)


# In[11]:


#df['NOTA_ENEM'] = 
pd.to_numeric(df.NOTA_ENEM, errors='coerce')
df.isnull().sum()


# In[12]:


NAN_NOTA_ENEM=df[np.isnan(df['NOTA_ENEM'])]


# In[13]:


NAN_NOTA_ENEM
df["POSSUI_FIES"]


# In[14]:


df["POSSUI_FIES"]= df["POSSUI_FIES"].replace({'SIM':"1" ,'NAO':"0"})
df.head()


# In[15]:


df["POSSUI_FIES"][df["POSSUI_FIES"]=="0"].groupby(by=df["SEXO"]).count()


# In[16]:


df["POSSUI_FIES"][df["POSSUI_FIES"]=="1"].groupby(by=df["SEXO"]).count()


# In[17]:


plt.figure(figsize=(6, 6))
labels =["Possui Fies","Não possui Fies"]
values = [26129,42697]
labels_gender = ["M","H","M","H"]
sizes_gender = [939,930 , 2544,2619]
colors = ['#ff6666', '#66b3ff']
colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
explode = (0.3,0.3) 
explode_gender = (0.1,0.1,0.1,0.1)
textprops = {"fontsize":15}
#Plot
plt.pie(values, labels=labels,autopct='%1.1f%%',pctdistance=1.08, labeldistance=0.8,colors=colors, startangle=90,frame=True, explode=explode,radius=10, textprops =textprops, counterclock = True, )
plt.pie(sizes_gender,labels=labels_gender,colors=colors_gender,startangle=90, explode=explode_gender,radius=7, textprops =textprops, counterclock = True, )
#Draw circle
centre_circle = plt.Circle((0,0),5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('COMPARAÇÃO DE GÊNERO QUE POSSUI FIES', fontsize=15, y=1.1)

# show plot 
 
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[18]:


df["SEMESTRES_CURSADOS"][0]>df['DURACAO_CURSO'][0]


# In[19]:


df.shape[0]


# In[20]:


type("SEMESTRES_CURSADOS")


# In[21]:


def target(df): 
    list_temp = []
    for i in range(400,500):
        if df['SEMESTRES_CURSADOS'][i] > df['DURACAO_CURSO'][i]:
            list_temp.append(0)
        elif df['SEMESTRES_CURSADOS'][i] <= df['DURACAO_CURSO'][i]:
            list_temp.append(1)
        else:
            list_temp.append(2)
    return list_temp
        


# In[22]:


def reprovado(df):
    if df['SEMESTRES_CURSADOS'] > df['DURACAO_CURSO']:
        return 0
    elif df['DURACAO_CURSO'] == df['SEMESTRES_CURSADOS']:
        return 1
    
NOVA_COLUNA=df.apply(reprovado, axis=1)
print(NOVA_COLUNA)

#new_df=df.assign("NOVA_COLUNA")


# In[24]:


df['NOVA_COLUNA'] = NOVA_COLUNA
df.head()
#new_df=df.assign("NOVA_COLUNA")


# In[25]:


df["NOVA_COLUNA"].value_counts(dropna=False)


# In[26]:


df.isnull().sum()


# In[27]:


df.dropna(subset=['NOVA_COLUNA'], inplace=True)
#EXCLUINDO VALORES 36 LINHAS COM VALORES NULOS NA NOVA_COLUNA, pois esta coluna é a que revela se o aluno terminou no tempo previsto o curso


# In[28]:


df.isnull().sum()


# In[29]:


df['NOVA_COLUNA'].value_counts(dropna=True)


# In[30]:


labels = df['NOVA_COLUNA'].unique()
values = df['NOVA_COLUNA'].value_counts()

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_layout(title_text="<b>DISTRIBUIÇÃO DE FORMADOS NO TEMPO PREVISTO</b>")
fig.show()


# In[31]:


sns.boxplot(x=df["T_IDADE"])
plt.title("BOX PLOT IDADE")
plt.grid(color="black",linestyle=":", linewidth=0.5)
#aplicando o conceito de box plot para verificar possíveis Outliers ma variável idade.


# In[32]:


df.columns


# ### Fazendo escolhas de colunas que podem interferir no atraso para conclusão do curso
# 
# # COLUNAS EXCLUÍDAS
# 1)COD_UNIDADE e COD_TURMA: Essas duas variáveis podem ser utilizadas em estudos futuros, para investigar, por exemplo, por que "tal" unidade tem menos alunos com atraso na conclusão do curso?
# 
# 2)DATA_NASCIMENTO: mostra excesso de informação, visto que existe uma coluna com idade.
# 
# 3)CIDADE_ALUNO, UF_ALUNO e CEP_ALUNO: ambas informações podem ser usadas em conjunto com o item ') para investigar quais condições da cidade ou estado favorem ou não para a conclusão sem atraso.
# 
# 4)COD_CURSO: pode ser excluida devido a utilização da coluna "NOME_CURSO", informações redundantes.
# 
# 5)SITUACAO_MATRICULA: uma coluna que mostra que todos estão formados, ou seja, nao acrescenta em nada para a análise.
# 
# 6)DATA_CONCLUSÃO: Não existe uma coluna "INICIO_CURSO" para comparação, ou seja, a data de conclusão não pode ser usada para analise de atraso, visto que não se sabe a data do inicio do curso.
# 
# 7)CH_TOTAL_MATRIZ, NR_TOTAL_DISCIPLINAS, CH_APROVADA e NR_DISC_APROVADA: São colunas que não fornecem comparações, visto que as informações contidas nela não elucida se o aluno foi ou não reprovado em alguma discilplina e consequentemente atrasando o curso.
# 
# 8)COD_ALUNO: Identificador único para cada alunos, sem número inteiro, ele pode atrapalhar os modelos.
# 
# 9)ESTADO_CIVIL: o estado civil do estudante interefere na conclusão ou não do curso no tempo correto?
# 
# 10)NOME_CURSO: Quais cursos tem o maior indice de reprovação? O que deve ser feito para sanar essa situação?
# 
# 11)NRO_REPRO_ACO, NRO_TOTAL_REPRO, NRO_REPRO_NORMAL: Comparar as reprovações é intuitivo para saber se concluiu ou nao no tempo estipulado.
# 
# 12)POSSUI_FIES: Alunos que possuem financiamento estudantil tende a ser mais aplicados? Isto é, terminam o curso no tempo estipulado?
# 
# # COLUNAS ESCOLHIDAS
# 
# 1)SEXO: o genêro interfere na conclusão ou não do curso no tempo correto?
# 
# 2)TURNO_CURSO: Os alunos que estudam a noite reprovam mais? Alunos de tempo integral? São perguntas que esperamos que o modelo responda.
# 
# 3)DURACAO_CURSO e SEMESTRES_CURSADOS: são as duas colunas chaves para a analise dos dados, visto que se a diferente entre seus itens forem maiores que 1, mostra que o aluno terminou o curso com atraso.
# 
# 4)PERIODOS_TRANCADOS: é intuitivo imaginar que se o aluno trancar o curso, ele não vai conseguir cumprir o curso nos semestres propostos.
# 
# 5)T_IDADE: qual a importância da idade na conclusão ou não do curso no tempo estipulado?
# 
# 6)NOTA: Notas boas são fatores predominantes para o aluno concluir no temopo estipulado o curso?
# 
# 7)NOTA_ENEM e POSSUI_ENEM: são variáveis que podem ser usadas para, por exemplo, investigar se alunos que possuem notas acima de "x" tem mais chances de concluir o curso no tempo estipulado.
# 
# 
# 
# ### AS ESCOLHAS PARA APLICAÇÃO DOS MODELOS FOI FEITA PARA OTIMIZAR O TEMPO, VISTO QUE, SERIA NECESSÁRIO MAIS TRATAMENTOS NOS DADOS FORNECIDOS PARA MELHORAR O MODELO.
# 
# 
# 
# 

# In[33]:


pri_trat=df[['SEXO', 'TURNO_CURSO', 'DURACAO_CURSO', 'PERIODOS_TRANCADOS', 'T_IDADE',
       'NOTA', 'SEMESTRES_CURSADOS','NOVA_COLUNA']]
#escolha das variáveis pertinentes para o  estudo de caso.


# In[34]:


pri_trat


# In[35]:


pri_trat.info()


# In[227]:


#num_cols = ["NOTA", 'T_IDADE']


# In[228]:


#df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')),columns=num_cols)


# In[241]:


#df_std


# In[233]:


#pri_trat["NOTA"]=df_std["NOTA"]
#pri_trat["T_IDADE"]=df_std["T_IDADE"]


# In[36]:


pri_trat


# # PROCESSAMENTO

# In[37]:


df=pri_trat
X = df.drop(columns = ['NOVA_COLUNA'])
y = df['NOVA_COLUNA'].values


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 40, stratify=y)


# # KNN

# In[50]:


knn_model = KNeighborsClassifier(n_neighbors = 11) 
knn_model.fit(X_train,y_train)
predicted_y = knn_model.predict(X_test)
accuracy_knn = knn_model.score(X_test,y_test)
print("KNN accuracy:",accuracy_knn)


# In[51]:


print(classification_report(y_test, predicted_y))


# # svc 

# In[52]:


svc_model = SVC(random_state = 1)
svc_model.fit(X_train,y_train)
predict_y = svc_model.predict(X_test)
accuracy_svc = svc_model.score(X_test,y_test)
print("SVM accuracy is :",accuracy_svc)


# In[53]:


print(classification_report(y_test, predict_y))


# # RANDON FOREST
# 

# In[54]:


model_rf = RandomForestClassifier(n_estimators=500 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))


# In[55]:


print(classification_report(y_test, prediction_test))


# In[56]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, prediction_test),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title(" RANDOM FOREST CONFUSION MATRIX",fontsize=14)
plt.show()

REGRESSÃO LINEAR
# In[57]:


lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
accuracy_lr = lr_model.score(X_test,y_test)
print("Logistic Regression accuracy is :",accuracy_lr)


# In[58]:


lr_pred= lr_model.predict(X_test)
report = classification_report(y_test,lr_pred)
print(report)


# In[59]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, lr_pred),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("LOGISTIC REGRESSION CONFUSION MATRIX",fontsize=14)
plt.show()


# In[ ]:




