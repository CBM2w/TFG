# -*- coding: utf-8 -*-
# +
import sys
sys.path.append("..")

from dataset.dataset_padchest import *


# -

def split_age(dataset, age):
    """
    Split input dataset stratifying patients by age

    :param Dataset dataset: Dataset containing the cases. Mandatory to contain
           X, y, weight, study_id, patient_idid, study_date, age and sex for train and val
    :param int age: Value from where to split cases between young and old people
    :return: Two datasets split by patient age
    """
    
    dataset_old = Dataset()
    dataset_young = Dataset()
    
    # Filter train dataset
    for i, case in enumerate(dataset.X_train):
        if dataset.age_train[i]>=age:
            dataset_old.X_train.append(case)
            dataset_old.y_train.append(dataset.y_train[i])
            dataset_old.weight_train.append(dataset.weight_train[i])
            dataset_old.study_id_train.append(dataset.study_id_train[i])
            dataset_old.patient_id_train.append(dataset.patient_id_train[i])
            dataset_old.study_date_train.append(dataset.study_date_train[i])
            dataset_old.age_train.append(dataset.age_train[i])
            dataset_old.sex_train.append(dataset.sex_train[i])
        else:
            dataset_young.X_train.append(case)
            dataset_young.y_train.append(dataset.y_train[i])
            dataset_young.weight_train.append(dataset.weight_train[i])
            dataset_young.study_id_train.append(dataset.study_id_train[i])
            dataset_young.patient_id_train.append(dataset.patient_id_train[i])
            dataset_young.study_date_train.append(dataset.study_date_train[i])
            dataset_young.age_train.append(dataset.age_train[i])
            dataset_young.sex_train.append(dataset.sex_train[i])
            
    for i, case in enumerate(dataset.X_val):
        if dataset.age_val[i]>=age:
            dataset_old.X_val.append(case)
            dataset_old.y_val.append(dataset.y_val[i])
            dataset_old.weight_val.append(dataset.weight_val[i])
            dataset_old.study_id_val.append(dataset.study_id_val[i])
            dataset_old.patient_id_val.append(dataset.patient_id_val[i])
            dataset_old.study_date_val.append(dataset.study_date_val[i])
            dataset_old.age_val.append(dataset.age_val[i])
            dataset_old.sex_val.append(dataset.sex_val[i])
        else:
            dataset_young.X_val.append(case)
            dataset_young.y_val.append(dataset.y_val[i])
            dataset_young.weight_val.append(dataset.weight_val[i])
            dataset_young.study_id_val.append(dataset.study_id_val[i])
            dataset_young.patient_id_val.append(dataset.patient_id_val[i])
            dataset_young.study_date_val.append(dataset.study_date_val[i])
            dataset_young.age_val.append(dataset.age_val[i])
            dataset_young.sex_val.append(dataset.sex_val[i])
    
    dataset_old.X_train = np.array(dataset_old.X_train)
    dataset_old.X_val = np.array(dataset_old.X_val)
    dataset_young.X_train = np.array(dataset_young.X_train)
    dataset_young.X_val = np.array(dataset_young.X_val)
    
    return dataset_old, dataset_young


def split_sex(dataset):
    """
    Split input dataset stratifying patients by sex

    :param Dataset dataset: Dataset containing the cases. Mandatory to contain
           X, y, weight, study_id, patient_id, study_date, age and sex for train and val
    :return: Two datasets split by patient sex
    """
    
    dataset_male = Dataset()
    dataset_female = Dataset()
    dataset_other = Dataset()
    
    # Filter train dataset
    for i, case in enumerate(dataset.X_train):
        if dataset.sex_train[i]=='M':
            dataset_male.X_train.append(case)
            dataset_male.y_train.append(dataset.y_train[i])
            dataset_male.weight_train.append(dataset.weight_train[i])
            dataset_male.study_id_train.append(dataset.study_id_train[i])
            dataset_male.patient_id_train.append(dataset.patient_id_train[i])
            dataset_male.study_date_train.append(dataset.study_date_train[i])
            dataset_male.age_train.append(dataset.age_train[i])
            dataset_male.sex_train.append(dataset.sex_train[i])
        elif dataset.sex_train[i]=='F':
            dataset_female.X_train.append(case)
            dataset_female.y_train.append(dataset.y_train[i])
            dataset_female.weight_train.append(dataset.weight_train[i])
            dataset_female.study_id_train.append(dataset.study_id_train[i])
            dataset_female.patient_id_train.append(dataset.patient_id_train[i])
            dataset_female.study_date_train.append(dataset.study_date_train[i])
            dataset_female.age_train.append(dataset.age_train[i])
            dataset_female.sex_train.append(dataset.sex_train[i])
        else:
            dataset_other.X_train.append(case)
            dataset_other.y_train.append(dataset.y_train[i])
            dataset_other.weight_train.append(dataset.weight_train[i])
            dataset_other.study_id_train.append(dataset.study_id_train[i])
            dataset_other.patient_id_train.append(dataset.patient_id_train[i])
            dataset_other.study_date_train.append(dataset.study_date_train[i])
            dataset_other.age_train.append(dataset.age_train[i])
            dataset_other.sex_train.append(dataset.sex_train[i])
            
    for i, case in enumerate(dataset.X_val):
        if dataset.sex_train[i]=='M':
            dataset_male.X_val.append(case)
            dataset_male.y_val.append(dataset.y_val[i])
            dataset_male.weight_val.append(dataset.weight_val[i])
            dataset_male.study_id_val.append(dataset.study_id_val[i])
            dataset_male.patient_id_val.append(dataset.patient_id_val[i])
            dataset_male.study_date_val.append(dataset.study_date_val[i])
            dataset_male.age_val.append(dataset.age_val[i])
            dataset_male.sex_val.append(dataset.sex_val[i])
        elif dataset.sex_train[i]=='F':
            dataset_female.X_val.append(case)
            dataset_female.y_val.append(dataset.y_val[i])
            dataset_female.weight_val.append(dataset.weight_val[i])
            dataset_female.study_id_val.append(dataset.study_id_val[i])
            dataset_female.patient_id_val.append(dataset.patient_id_val[i])
            dataset_female.study_date_val.append(dataset.study_date_val[i])
            dataset_female.age_val.append(dataset.age_val[i])
            dataset_female.sex_val.append(dataset.sex_val[i])
        else:
            dataset_other.X_val.append(case)
            dataset_other.y_val.append(dataset.y_val[i])
            dataset_other.weight_val.append(dataset.weight_val[i])
            dataset_other.study_id_val.append(dataset.study_id_val[i])
            dataset_other.patient_id_val.append(dataset.patient_id_val[i])
            dataset_other.study_date_val.append(dataset.study_date_val[i])
            dataset_other.age_val.append(dataset.age_val[i])
            dataset_other.sex_val.append(dataset.sex_val[i])
        
    dataset_male.X_train = np.array(dataset_male.X_train)
    dataset_male.X_val = np.array(dataset_male.X_val)
    dataset_female.X_train = np.array(dataset_female.X_train)
    dataset_female.X_val = np.array(dataset_female.X_val)
    dataset_other.X_train = np.array(dataset_other.X_train)
    dataset_other.X_val = np.array(dataset_other.X_val)
            
    return dataset_male, dataset_female, dataset_other


def delete_label(dataset, label):
    """
    Delete cases containing specific label

    :param Dataset dataset: Dataset containing the cases. Mandatory to contain
           X, y, weight, study_id, patient_id, study_date, age and sex for train and val
    :param string label: Label of the cases being ignored
    :return: Two datasets split by patient sex
    """
    
    new_dataset = Dataset()
    
    # Filter train dataset
    for i, case in enumerate(dataset.y_train):
        if dataset.labels[np.argmax(case)] != label:
            new_dataset.X_train.append(dataset.X_train[i])
            new_dataset.y_train.append(case)
            new_dataset.weight_train.append(dataset.weight_train[i])
            new_dataset.study_id_train.append(dataset.study_id_train[i])
            new_dataset.patient_id_train.append(dataset.patient_id_train[i])
            new_dataset.study_date_train.append(dataset.study_date_train[i])
            new_dataset.age_train.append(dataset.age_train[i])
            new_dataset.sex_train.append(dataset.sex_train[i])
            
    for i, case in enumerate(dataset.y_val):
        if dataset.labels[np.argmax(case)] != label:
            new_dataset.X_val.append(dataset.X_val[i])
            new_dataset.y_val.append(case)
            new_dataset.weight_val.append(dataset.weight_val[i])
            new_dataset.study_id_val.append(dataset.study_id_val[i])
            new_dataset.patient_id_val.append(dataset.patient_id_val[i])
            new_dataset.study_date_val.append(dataset.study_date_val[i])
            new_dataset.age_val.append(dataset.age_val[i])
            new_dataset.sex_val.append(dataset.sex_val[i])
        
    new_dataset.X_train = np.array(new_dataset.X_train)
    new_dataset.X_val = np.array(new_dataset.X_val)
            
    return new_dataset
