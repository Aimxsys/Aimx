# Audex

FROM python:3.8

# Pull tensorflow image with Python3
FROM tensorflow/tensorflow:2.1.0-py3

# Set the working directory to /Aimx
WORKDIR /Aimx

# Create all the necessary directories in the container
RUN mkdir /Aimx/Audex
RUN mkdir /Aimx/Audex/utils
RUN mkdir /Aimx/workdir
RUN mkdir /Aimx/workdir/gen_models
RUN mkdir /Aimx/workdir/gen_models/model_cnn_e50_2v_977d_speech_commands_v001_13m_2048w_512h_5i_22050r_1s

# Transfer functionality content from current dir to the corresponding dir in the container
ADD ./__init__.py           /Aimx/Audex
ADD ./app_server.py         /Aimx/Audex
ADD ./dataprep_asr.py       /Aimx/Audex
ADD ./dataprep_genre.py     /Aimx/Audex
ADD ./service_wordetect.py  /Aimx/Audex
ADD ./train_asr_cnn.py      /Aimx/Audex
ADD ./train_genre_ann.py    /Aimx/Audex
ADD ./train_genre_cnn.py    /Aimx/Audex
ADD ./train_genre_rnn.py    /Aimx/Audex

ADD ./utils/__init__.py     /Aimx/Audex/utils
ADD ./utils/utils_audex.py  /Aimx/Audex/utils
ADD ./utils/utils_common.py /Aimx/Audex/utils

# Transfer accompanying resources from current dir to the corresponding dir in the container
ADD ./docker_resources/requirements_linux.txt /Aimx
ADD ./docker_resources/uwsgi_server.ini       /Aimx

# Transfer the model with which to do inference
ADD   ./docker_resources/model_cnn_e50_2v_977d_speech_commands_v001_13m_2048w_512h_5i_22050r_1s \
/Aimx/workdir/gen_models/model_cnn_e50_2v_977d_speech_commands_v001_13m_2048w_512h_5i_22050r_1s

# Transfer the metadata files
ADD ./docker_resources/dataprep_result_meta.json /Aimx/workdir
ADD ./docker_resources/training_result_meta.json /Aimx/workdir

# Install tree
RUN apt-get install tree

# See the structure created by this point
RUN pwd
RUN tree -L 4

# Install audio libraries required, in particular, by librosa
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# First, upgrade pip
RUN pip install --upgrade pip

# Install python packages, "--no-cache-dir" seemingly fixed a hang on downloading tensorflow
RUN pip install -r requirements_linux.txt --no-cache-dir 

# Start uWSGI using config file
CMD ["uwsgi", "uwsgi_server.ini"]