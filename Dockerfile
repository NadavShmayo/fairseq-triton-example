FROM nvcr.io/nvidia/tritonserver:24.09-py3

RUN pip install pip==24.0
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install fairseq==0.12.2
