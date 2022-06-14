print("experiments")



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn
import pickle
#from your_data_loader import load_data
from churnexplainer import CategoricalEncoder
import datetime

import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import xml.etree.ElementTree as ET
from cmlbootstrap import CMLBootstrap
# Set the setup variables needed by CMLBootstrap
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)
runtimes=cml.get_runtimes()
runtimes=runtimes['runtimes']
runtimesdf = pd.DataFrame.from_dict(runtimes, orient='columns')
runtimeid=runtimesdf.loc[(runtimesdf['editor'] == 'Workbench') & (runtimesdf['kernel'] == 'Python 3.7') & (runtimesdf['edition'] == 'Standard')]['id']
id_rt=runtimeid.values[0]

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .getOrCreate()

# **Note:**
# Our file isn't big, so running it in Spark local mode is fine but you can add the following config
# if you want to run Spark on the kubernetes cluster
#
# > .config("spark.yarn.access.hadoopFileSystems",os.getenv['STORAGE'])\
#
# and remove `.master("local[*]")\`
#

# Since we know the data already, we can add schema upfront. This is good practice as Spark will
# read *all* the Data if you try infer the schema.

schema = StructType(
    [
        StructField("customerID", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("SeniorCitizen", StringType(), True),
        StructField("Partner", StringType(), True),
        StructField("Dependents", StringType(), True),
        StructField("tenure", DoubleType(), True),
        StructField("PhoneService", StringType(), True),
        StructField("MultipleLines", StringType(), True),
        StructField("InternetService", StringType(), True),
        StructField("OnlineSecurity", StringType(), True),
        StructField("OnlineBackup", StringType(), True),
        StructField("DeviceProtection", StringType(), True),
        StructField("TechSupport", StringType(), True),
        StructField("StreamingTV", StringType(), True),
        StructField("StreamingMovies", StringType(), True),
        StructField("Contract", StringType(), True),
        StructField("PaperlessBilling", StringType(), True),
        StructField("PaymentMethod", StringType(), True),
        StructField("MonthlyCharges", DoubleType(), True),
        StructField("TotalCharges", DoubleType(), True),
        StructField("Churn", StringType(), True)
    ]
)

# Now we can read in the data from Cloud Storage into Spark...
try : 
  storage=os.environ["STORAGE"]
except:
  if os.path.exists("/etc/hadoop/conf/hive-site.xml"):
    tree = ET.parse('/etc/hadoop/conf/hive-site.xml')
    root = tree.getroot()
    for prop in root.findall('property'):
      if prop.find('name').text == "hive.metastore.warehouse.dir":
        storage = prop.find('value').text.split("/")[0] + "//" + prop.find('value').text.split("/")[2]
  else:
    storage = "/user/" + os.getenv("HADOOP_USER_NAME")
  storage_environment_params = {"STORAGE":storage}
  storage_environment = cml.create_environment_variable(storage_environment_params)
  os.environ["STORAGE"] = storage


storage = os.environ['STORAGE']
hadoop_user = os.environ['HADOOP_USER_NAME']

telco_data = spark.read.csv(
    "{}/user/{}/data/churn/WA_Fn-UseC_-Telco-Customer-Churn-.csv".format(
        storage,hadoop_user),
    header=True,
    schema=schema,
    sep=',',
    nullValue='NA'
)

# ...and inspect the data.

telco_data.show()

telco_data.printSchema()

# Now we can store the Spark DataFrame as a file in the local CML file system
# *and* as a table in Hive used by the other parts of the project.

telco_data.coalesce(1).write.csv(
    "file:/home/cdsw/raw/telco-data/",
    mode='overwrite',
    header=True
)

spark.sql("show databases").show()

spark.sql("show tables in default").show()

# Create the Hive table
# This is here to create the table in Hive used be the other parts of the project, if it
# does not already exist.

if ('telco_churn' not in list(spark.sql("show tables in default").toPandas()['tableName'])):
    print("creating the telco_churn database")
    telco_data\
        .write.format("parquet")\
        .mode("overwrite")\
        .saveAsTable(
            'default.telco_churn'
        )

# Show the data in the hive table
spark.sql("select * from default.telco_churn").show()

# To get more detailed information about the hive table you can run this:
df = spark.sql("SELECT * FROM default.telco_churn").toPandas()


idcol = 'customerID'
labelcol = 'Churn'
cols = (('gender', True),
        ('SeniorCitizen', True),
        ('Partner', True),
        ('Dependents', True),
        ('tenure', False),
        ('PhoneService', True),
        ('MultipleLines', True),
        ('InternetService', True),
        ('OnlineSecurity', True),
        ('OnlineBackup', True),
        ('DeviceProtection', True),
        ('TechSupport', True),
        ('StreamingTV', True),
        ('StreamingMovies', True),
        ('Contract', True),
        ('PaperlessBilling', True),
        ('PaymentMethod', True),
        ('MonthlyCharges', False),
        ('TotalCharges', False))


df = df.replace(r'^\s$', np.nan, regex=True).dropna().reset_index()
df.index.name = 'id'
data, labels = df.drop(labelcol, axis=1), df[labelcol]
data = data.replace({'SeniorCitizen': {1: 'Yes', 0: 'No'}})
# This is Mike's lovely short hand syntax for looping through data and doing useful things. I think if we started to pay him by the ASCII char, we'd get more readable code.
data = data[[c for c, _ in cols]]
catcols = (c for c, iscat in cols if iscat)
for col in catcols:
    data[col] = pd.Categorical(data[col])
labels = (labels == 'Yes')

ce = CategoricalEncoder()
X = ce.fit_transform(data)


y=labels.values
print("empenzando los experimentos")

run_time_suffix = datetime.datetime.now()
#run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")
run_time_suffix = run_time_suffix.strftime("%M%S")
#X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if len(sys.argv) == 2:
    try:
        a=mlflow.get_run(sys.argv[1]).data.params
        print(a)
        n_estimators = int(a["n_estimators"])
        rf = RandomForestRegressor(n_estimators=n_estimators)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
 
        filename = './models/champion/ce.pkl'
        pickle.dump(ce, open(filename, 'wb'))

        filename = './models/champion/champion.pkl'
        pickle.dump(rf, open(filename, 'wb'))

       
        project_id = cml.get_project()['id']
        params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}


        default_engine_details = cml.get_default_engine({})
        default_engine_image_id = default_engine_details["id"]


        print("creando el modelo")

        example_model_input = {"StreamingTV": "No", "MonthlyCharges": 70.35, "PhoneService": "No", "PaperlessBilling": "No", "Partner": "No", "OnlineBackup": "No", "gender": "Female", "Contract": "Month-to-month", "TotalCharges": 1397.475,
                       "StreamingMovies": "No", "DeviceProtection": "No", "PaymentMethod": "Bank transfer (automatic)", "tenure": 29, "Dependents": "No", "OnlineSecurity": "No", "MultipleLines": "No", "InternetService": "DSL", "SeniorCitizen": "No", "TechSupport": "No"}


        try:
          
                    
                      # Create the YAML file for the model lineage
            yaml_text = \
                """"ModelOpsChurn":
              hive_table_qualified_names:                # this is a predefined key to link to training data
                - "default.telco_churn@cm"               # the qualifiedName of the hive_table object representing                
              metadata:                                  # this is a predefined key for additional metadata
                query: "select * from historical_data"   # suggested use case: query used to extract training data
                training_file: "3_trainStrategy_job.py"       # suggested use case: training file used
            """

            with open('lineage.yml', 'w') as lineage:
                lineage.write(yaml_text)
            model_id = cml.get_models(params)[0]['id']
            latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

            build_model_params = {
              "modelId": latest_model['latestModelBuild']['modelId'],
              "projectId": latest_model['latestModelBuild']['projectId'],
              "targetFilePath": "11_best_model_serve.py",
              "targetFunctionName": "explain",
              "engineImageId": default_engine_image_id,
              "kernel": "python3",
              "examples": latest_model['latestModelBuild']['examples'],
              "cpuMillicores": 1000,
              "memoryMb": 2048,
              "nvidiaGPUs": 0,
              "replicationPolicy": {"type": "fixed", "numReplicas": 1},
              "environment": {},"runtimeId":int(id_rt)}

            cml.rebuild_model(build_model_params)
            sys.argv=[]
            print('rebuilding...')
            # Wait for the model to deploy.

          
        except:
          
                      # Create the YAML file for the model lineage
            yaml_text = \
                """"ModelOpsChurn":
              hive_table_qualified_names:                # this is a predefined key to link to training data
                - "default.telco_churn@cm"               # the qualifiedName of the hive_table object representing                
              metadata:                                  # this is a predefined key for additional metadata
                query: "select * from historical_data"   # suggested use case: query used to extract training data
                training_file: "3_trainStrategy_job.py"       # suggested use case: training file used
            """

            with open('lineage.yml', 'w') as lineage:
                lineage.write(yaml_text)

            create_model_params = {
                "projectId": project_id,
                "name": "ModelOpsChurn",
                "description": "Explain a given model prediction",
                "visibility": "private",
                "enableAuth": False,
                "targetFilePath": "11_best_model_serve.py",
                "targetFunctionName": "explain",
                "engineImageId": default_engine_image_id,
                "kernel": "python3",
                "examples": [
                    {
                        "request": example_model_input,
                        "response": {}
                    }],
                "cpuMillicores": 1000,
                "memoryMb": 2048,
                "nvidiaGPUs": 0,
                "replicationPolicy": {"type": "fixed", "numReplicas": 1},
                "environment": {},"runtimeId":int(id_rt)}
            print("creando nuevo modelo")
            new_model_details = cml.create_model(create_model_params)
            access_key = new_model_details["accessKey"]  # todo check for bad response
            model_id = new_model_details["id"]

            print("New model created with access key", access_key)

            # Disable model_authentication
            cml.set_model_auth({"id": model_id, "enableAuth": False})
            sys.argv=[]

            # Wait for the model to deploy.
            is_deployed = False
            while is_deployed == False:
                model = cml.get_model({"id": str(
                    new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True})
                if model["latestModelDeployment"]["status"] == 'deployed':
                    print("Model is deployed")
                    break
                else:
                    print("Deploying Model.....")
                    time.sleep(10)


    except:
        sys.exit("Invalid Arguments passed to Experiment")
        sys.argv=[]
else:
    try:
      experimentId=mlflow.get_experiment_by_name("expRetrain").experiment_id
      mlflow.delete_experiment(experimentId)

      time.sleep(20)
    except:
      print("es la primera vez que se ejecuta")




    #mlflow.set_tracking_uri('http://your.mlflow.url:5000')
    mlflow.set_experiment('expRetrain')
    valuesParam=[9,11,15]
    for i in range(len(valuesParam)):
      with mlflow.start_run(run_name="run_"+run_time_suffix+'_'+str(i)) as run: 
          # tracking run parameters
          mlflow.log_param("compute", 'local')
          mlflow.log_param("dataset", 'telco-churn')
          mlflow.log_param("dataset_version", '2.0')
          mlflow.log_param("algo", 'random forest')

          # tracking any additional hyperparameters for reproducibility
          n_estimators = valuesParam[i]
          mlflow.log_param("n_estimators", n_estimators)

          # train the model
          rf = RandomForestRegressor(n_estimators=n_estimators)
          rf.fit(X_train, y_train)
          y_pred = rf.predict(X_test)

          # automatically save the model artifact to the S3 bucket for later deployment
          mlflow.sklearn.log_model(rf, "rf-baseline-model")

          # log model performance using any metric
          precision=average_precision_score(y_test, y_pred)
          #mse = mean_squared_error(y_test, y_pred)
          mlflow.log_metric("precision", precision)

          mlflow.end_run()


    
