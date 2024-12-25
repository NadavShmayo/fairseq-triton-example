### Steps to run
1. Download fairseq `transformer.wmt14.en-fr` from this URL: https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2
2. Extract the bz2 and tar archives into the root directory of the project
3. Install the project requirements (`pip install -r requirements.txt`)
4. Run `compile_decoder.py` and move the output `model.pt` file to `models/decoder/1/model.pt`
5. Run `compile_enncoder.py` and move the output `model.pt` file to `models/encoder/1/model.pt`
6. Build the Docker image using the provided Dockerfile (`docker build .`)
7. Run the built Docker image with the compiled models, the models directory should be mounted to `/mnt/models` in the container
8. Run locust using `locust` CLI command and open the browser to `http://localhost:8089` to start the test
9. Run the test with host configured to the Tritonserver we ran, 15 peak concurrency, and 5 spawn rate