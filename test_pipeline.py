import subprocess

subprocess.call(['genre_preprocess.py', '-dataset_path', 'dataset_c10_f3'], shell=True)

subprocess.call(['genre_classifier.py', '-data_path', 'most_recent_output', '-epochs', '5'], shell=True)