from locust import HttpUser, task
from locust.exception import LocustError


class TritonTranslationUser(HttpUser):
    @task
    def translate(self):
        body = {
            "inputs": [
                {
                    "name": "src_tokens",
                    "shape": [1, 4],
                    "datatype": "INT64",
                    "data": [[134, 16, 65, 2]]
                },
                {
                    "name": "src_lengths",
                    "shape": [1, 1],
                    "datatype": "INT64",
                    "data": [[4]]
                }
            ]
        }
        response_json = self.client.post("/v2/models/bls/infer", json=body).json()
        expected_result = [134, 16, 65, 2]
        if response_json["outputs"][0]["data"] != expected_result:
            print(response_json["outputs"][0]["data"])
