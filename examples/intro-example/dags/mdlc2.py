# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
### Tutorial Documentation
Documentation that goes along with the Airflow tutorial located
[here](https://airflow.incubator.apache.org/tutorial.html)
"""
from datetime import timedelta

import airflow
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from ml_framework.preprocess import preprocess
from ml_framework.comparer import calculateScore
from ml_framework.split_and_validate import score
from airflow.models import Variable



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': airflow.utils.dates.days_ago(1),
    'email': ['tabish.maniar@synechron.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
    # 'concurrency': 2,
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'adhoc':False,
    # 'sla': timedelta(hours=2),
    'execution_timeout': timedelta(seconds=50000)
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'trigger_rule': u'all_success'
}


pred_column = Variable.get("pred_column")
pred_values = Variable.get("pred_values").split(",")
file_path = Variable.get("file_path")
delimiter = Variable.get("delimiter")
one_hot_columns = Variable.get("one_hot_encoding_columns").split(",")
kfold_validation_count = int(Variable.get("kfold_validation_count"))
drop_columns = Variable.get("drop_columns",default_var="").split(",")

simplicity_hash = Variable.get("simplicity_hash",deserialize_json=True )
speed_hash = Variable.get("speed_hash",deserialize_json=True)
weightage_hash = Variable.get("weightage_hash",deserialize_json=True)

dag = DAG(
    'mdlc2',
    default_args=default_args,
    description='Machine Learning Pipeline Lifecycle',
    schedule_interval=timedelta(days=1),
)


t1 = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess,
    op_args=[kfold_validation_count,file_path,pred_column,pred_values,delimiter,one_hot_columns,drop_columns],
    dag=dag,
)

ml_algos = Variable.get("ml_algos").split(",")
heavy_algos = Variable.get("heavy_algos").split(",")
# ml_algos = ['neuralNetwork']

# naiveBayes,neuralNetwork,decisionTree,adaBoost,randomForestClassifier,logisticRegressionClassifier,qda,lda
# decisionTree','randomForestClassifier','logisticRegressionClassifier']

    # kwargs['ti'].xcom_push(key='load_cycle_id_3',value=44444)


# def add_dynamo(cls,i):
#     def innerdynamo(**kwargs):
#         print "in score %d" % i
#         model_score = score(i)

#     innerdynamo.__doc__ = "docstring for score_%d" % i
#     innerdynamo.__name__ = "score_%d" % i
#     setattr(cls,innerdynamo.__name__,innerdynamo)

kfoldTasks = []



def push_score(idx,**kwargs):
    model_score = score(idx)
    kwargs['ti'].xcom_push(key=f'score_{idx}',value=model_score)


def average_score(algo,**context):
    # print(context)
    final_score = context['ti'].xcom_pull(task_ids=f'score_0_{algo}')

    for i in range(1,kfold_validation_count):
        for (key,value) in  context['ti'].xcom_pull(task_ids=f'score_{i}_{algo}').items() :
            final_score[key] += value
    for (key,value) in final_score.items():
        final_score[key] /= kfold_validation_count
    return {algo: final_score}


def calculateModelScore(**context):
    print('comparing please wait')
    storeScores = {}
    for algo in ml_algos :
       storeScores.update(context['ti'].xcom_pull(task_ids=f'collate_final_score_{algo}'))
    print(storeScores)
    ratio = context['ti'].xcom_pull(task_ids='preprocess')
    calculateScore(storeScores,[],simplicity_hash,weightage_hash,speed_hash,ratio)


compare_final_score = PythonOperator(
    task_id = "compare_result",
    python_callable=calculateModelScore,
    provide_context=True,
    dag=dag
    )
prev_collate_score = None
for algo in ml_algos :
    collate_score = PythonOperator (
            task_id = f"collate_final_score_{algo}",
            python_callable=average_score,
            op_args=[algo],
            provide_context=True,
            dag = dag
        )
    prev_task_score = None
    for i in range(kfold_validation_count):
        task_score = PythonOperator(
            task_id = f'score_{i}_{algo}',
            python_callable=score,
            op_args=[i,algo],
            # provide_context=True,
            dag = dag
            )
        kfoldTasks.append(
            task_score
            )
        # t3 = BashOperator(
        #     task_id=f'print_date_{i}_{algo}',
        #     bash_command='sleep 1m',
        #     dag=dag)

        if algo in heavy_algos and prev_task_score != None:
            prev_task_score >> task_score

        if algo in heavy_algos and i == 0 :
            if prev_collate_score == None :
                t1 >> task_score
            else :
                prev_collate_score >> task_score
        elif algo not in heavy_algos :
            if prev_collate_score == None :
                t1 >> task_score
            else :
                prev_collate_score >> task_score
        prev_task_score = task_score
        if algo not in heavy_algos :
            task_score >> collate_score
        elif algo in heavy_algos and i == kfold_validation_count - 1 :
            task_score >> collate_score

    prev_collate_score = collate_score

    collate_score >> compare_final_score


