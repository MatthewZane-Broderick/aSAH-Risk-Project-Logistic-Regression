import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

num_features =[
    'age',
    'wfns_grade',
    'fisher_grade',
    'sbp_admission',
    'tmt_mean',
    'tma_mean',
    'bmi',
    'wbc',
    'lymphocyte_abs',
    'nlr',
    'albumin'
    'glucose',
    'aneurysm_size_mm']