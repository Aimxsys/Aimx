# Audex

# Pull the latest tensorflow image
FROM tensorflow/tensorflow

# Set the working directory to /Aimx
WORKDIR /Aimx

# Transfer Python requirements
COPY ./docker_resources/requirements_linux.txt /Aimx

# Install tree
RUN apt-get install tree

# Install audio libraries required, in particular, by librosa
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# First, upgrade pip
RUN pip install --upgrade pip

# Install python packages, "--no-cache-dir" seemingly fixed a hang on downloading tensorflow
RUN pip install -r requirements_linux.txt --no-cache-dir 

# Create all the necessary directories in the container
RUN mkdir /Aimx/Audex
RUN mkdir /Aimx/Audex/utils
RUN mkdir /Aimx/workdir
RUN mkdir /Aimx/workdir/gen_models
RUN mkdir /Aimx/workdir/gen_models/model_cnn_e50_2v_977d_speech_commands_v001_13m_2048w_512h_5i_22050r_1s

# Transfer accompanying resources from current dir to the corresponding dir in the container
COPY ./docker_resources/uwsgi_server.ini /Aimx/Audex

# Transfer functionality content from current dir to the corresponding dir in the container
COPY ./__init__.py           /Aimx/Audex
COPY ./app_server.py         /Aimx/Audex
COPY ./dataprep_asr.py       /Aimx/Audex
COPY ./dataprep_genre.py     /Aimx/Audex
COPY ./service_wordetect.py  /Aimx/Audex
COPY ./train_asr_cnn.py      /Aimx/Audex
COPY ./train_genre_ann.py    /Aimx/Audex
COPY ./train_genre_cnn.py    /Aimx/Audex
COPY ./train_genre_rnn.py    /Aimx/Audex

COPY ./utils/__init__.py     /Aimx/Audex/utils
COPY ./utils/utils_audex.py  /Aimx/Audex/utils
COPY ./utils/utils_common.py /Aimx/Audex/utils

# Transfer the model with which to do inference
COPY  ./docker_resources/model_cnn_e50_2v_977d_speech_commands_v001_13m_2048w_512h_5i_22050r_1s \
/Aimx/workdir/gen_models/model_cnn_e50_2v_977d_speech_commands_v001_13m_2048w_512h_5i_22050r_1s

# Transfer the metadata files
COPY ./docker_resources/dataprep_result_meta.json /Aimx/workdir
COPY ./docker_resources/training_result_meta.json /Aimx/workdir

# See the structure created by this point
RUN pwd
RUN tree -L 4

# Workdir must be set to Audex in order for the whole thing to run correctly
# (i.e. for paths like Audex's workdir to resolve to /Aimx/workdir, etc.)
WORKDIR /Aimx/Audex

# Start uWSGI using config file
CMD ["uwsgi", "uwsgi_server.ini"]