#!/bin/bash

# grep -E '/(chr4|chr10|chr16)/256/' dataset_dict_5000.txt > dataset_dict_5000_256.val
# grep -E '/(chr6|chr12|chr18)/256/' dataset_dict_5000.txt > dataset_dict_5000_256.test
# grep '/256/' dataset_dict_5000.txt | grep -vE '/(chr4|chr10|chr16|chr6|chr12|chr18)/' > dataset_dict_5000_256.train


# grep -E '/(dmso_[^/]+|dtag_[^/]+)/' dataset_dict_5000.txt | grep -E '/(chr4|chr10|chr16)/256/' > dataset_dict_5000_256.val
# grep -E '/(dmso_[^/]+|dtag_[^/]+)/' dataset_dict_5000.txt | grep -E '/(chr6|chr12|chr18)/256/' > dataset_dict_5000_256.test
# grep -E '/(dmso_[^/]+|dtag_[^/]+)/' dataset_dict_5000.txt | grep '/256/' | grep -vE '/(chr4|chr10|chr16|chr6|chr12|chr18)/' > dataset_dict_5000_256.train

grep -E '/(dmso_control_60m+|dtag_v1_60m+)/' dataset_dict_5000.txt | grep -E '/(chr4|chr10|chr16)/256/' > dataset_dict_5000_256.val
grep -E '/(dmso_control_60m+|dtag_v1_60m+)/' dataset_dict_5000.txt | grep -E '/(chr6|chr12|chr18)/256/' > dataset_dict_5000_256.test
grep -E '/(dmso_control_60m+|dtag_v1_60m+)/' dataset_dict_5000.txt | grep '/256/' | grep -vE '/(chr4|chr10|chr16|chr6|chr12|chr18)/' > dataset_dict_5000_256.train