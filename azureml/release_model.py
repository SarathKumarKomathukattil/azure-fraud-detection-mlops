import time
from azure.ai.ml import MLClient
from pathlib import Path
from azure.identity import DefaultAzureCredential


BLUE = "blue"
GREEN = "green"

CANARY_WAIT_SECONDS = 60

ENDPOINT_NAME = 'fraud-identify-endpoint'
MODEL_NAME = 'fraud-xgboost'
REQUEST_FILE = Path(__file__).resolve().parent/'sample_request.json'

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential,
                                 path=Path(__file__).resolve().parents[1]/'.azureml'/'config.json')


def find_challenger():
    challengers = []

    for model in ml_client.models.list(name=MODEL_NAME):
        tags = model.tags or {}

        if tags.get('quality_gate') == 'passed' and tags.get('release_status') == 'challenger':
            challengers.append(model)

    if not challengers:
        return None

    return max(challengers,key=lambda model: int(model.version))

def set_traffic(active_slot,active_percent,candidate_slot,candidate_percent):
    endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)

    endpoint.traffic = {
        active_slot:active_percent,
        candidate_slot:candidate_percent
    }

    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def smoke_test(deployment_name):
    response = ml_client.online_endpoints.invoke(endpoint_name=ENDPOINT_NAME,
                                                 request_file=str(REQUEST_FILE),
                                                 deployment_name=deployment_name,
                                                 )

    if not response:
        raise RuntimeError('Smoke test failed')

    print(f'Smoke test passed: {response}')


def main():
    if not REQUEST_FILE.exists():
        raise FileNotFoundError(f'Request file not found: {REQUEST_FILE}')

    challenger = find_challenger()

    if challenger is None:
        print('No approved challenger found')
        return

    challenger_version = str(challenger.version)
    print(f'Approved challenger: {MODEL_NAME}:{challenger_version}')

    endpoint =  ml_client.online_endpoints.get(ENDPOINT_NAME)
    traffic = endpoint.traffic or {}

    blue_traffic = int(
        traffic.get(BLUE, 0)
    )

    green_traffic = int(
        traffic.get(GREEN, 0)
    )

    if (
        blue_traffic == 100
        and green_traffic == 0
    ):

        active_slot = BLUE
        candidate_slot = GREEN

    elif (
        green_traffic == 100
        and blue_traffic == 0
    ):

        active_slot = GREEN
        candidate_slot = BLUE

    else:

        raise RuntimeError(
            "Endpoint must be in a clean "
            "100/0 state before deployment."
        )

    print(
        "Active slot:",
        active_slot
    )

    print(
        "Candidate slot:",
        candidate_slot
    )

    active_deployment = (
        ml_client.online_deployments.get(
            name=active_slot,
            endpoint_name=ENDPOINT_NAME,
        )
    )

    model_reference = str(
        active_deployment.model
    )

    if "/versions/" in model_reference:

        current_version = (
            model_reference
            .split("/versions/")[-1]
        )

    else:

        current_version = (
            model_reference
            .split(":")[-1]
        )

    print(
        "Current champion:",
        f"{MODEL_NAME}:{current_version}"
    )

    if challenger_version == current_version:

        print(
            "Model is already in production."
        )

        return

    candidate_deployment = (
        ml_client.online_deployments.get(
            name=candidate_slot,
            endpoint_name=ENDPOINT_NAME,
        )
    )

    candidate_deployment.model = (
        f"azureml:{MODEL_NAME}:"
        f"{challenger_version}"
    )

    print(
        "Deploying challenger to",
        candidate_slot
    )

    result = (
        ml_client.online_deployments
        .begin_create_or_update(
            candidate_deployment
        )
        .result()
    )

    if (
        result.provisioning_state
        != "Succeeded"
    ):

        raise RuntimeError(
            "Candidate deployment failed."
        )

    smoke_test(
        candidate_slot
    )

    try:

        print(
            "Starting 90/10 canary..."
        )

        set_traffic(
            active_slot,
            90,
            candidate_slot,
            10,
        )

        print(
            f"Waiting {CANARY_WAIT_SECONDS} "
            "seconds..."
        )

        time.sleep(
            CANARY_WAIT_SECONDS
        )

        smoke_test(
            candidate_slot
        )

        print(
            "Promoting challenger..."
        )

        set_traffic(
            active_slot,
            0,
            candidate_slot,
            100,
        )

    except Exception:

        print(
            "Release failed. "
            "Rolling back..."
        )

        set_traffic(
            active_slot,
            100,
            candidate_slot,
            0,
        )

        raise

    try:

        challenger_model = (
            ml_client.models.get(
                name=MODEL_NAME,
                version=challenger_version,
            )
        )

        challenger_model.tags = {
            **(challenger_model.tags or {}),
            "release_status": "champion",
        }

        ml_client.models.create_or_update(
            challenger_model
        )

        previous_model = (
            ml_client.models.get(
                name=MODEL_NAME,
                version=current_version,
            )
        )

        previous_model.tags = {
            **(previous_model.tags or {}),
            "release_status": "rollback",
        }

        ml_client.models.create_or_update(
            previous_model
        )

    except Exception as error:

        print(
            "Warning: deployment succeeded "
            "but model tags were not updated."
        )

        print(error)

    print("")
    print(
        "Production release successful."
    )

    print(
        "New champion:",
        f"{MODEL_NAME}:{challenger_version}"
    )

    print(
        "Production slot:",
        candidate_slot
    )

    print(
        "Rollback slot:",
        active_slot
    )


if __name__ == "__main__":
    main()





    