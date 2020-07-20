FROM continuumio/miniconda:latest

WORKDIR .

COPY boot.sh ./
COPY environment.yml ./
COPY app.py ./
COPY _config.py ./
COPY models.py ./
COPY output_folder/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json ./output_folder/
COPY inference ./inference/
COPY results ./results/
COPY trained_models ./trained_models/
COPY uploads ./uploads/

RUN chmod +x ./app.py
RUN chmod +x ./boot.sh
RUN chmod +x ./_config.py
RUN chmod +x ./models.py
RUN chmod +x ./inference/*
RUN chmod +x ./results/*
RUN chmod +x ./trained_models/*
RUN chmod +x ./uploads/*

RUN conda env create -f environment.yml
RUN echo "conda activate pytorch" > ~/.bashrc
ENV PATH /opt/conda/envs/pytorch/bin:$PATH

RUN python -c "import torch"

EXPOSE 5000
ENTRYPOINT ["bash", "./boot.sh"]
#CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]