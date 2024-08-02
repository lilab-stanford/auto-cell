import numpy as np
from core import Core
from dataset import Dataset
from pprint import pprint
from sklearn.metrics import confusion_matrix
import pickle


ds = Dataset('./Predicted Texts/')
ds.calculate_biomarker()
ds.save()
ds.calculate_biomarker_mean()
ds.univariate_cox_model()
ds.log_rank_test()
ds.save()
