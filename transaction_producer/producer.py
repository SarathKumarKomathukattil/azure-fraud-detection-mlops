import os
import uuid
import random
import logging

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.exceptions import ResourceExistsError


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(
    "fraud-batch-producer"
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ADLS_ACCOUNT_NAME = os.getenv(
    "ADLS_ACCOUNT_NAME",
    "fraudmlopsstorage"
)

ADLS_FILESYSTEM = os.getenv(
    "ADLS_FILESYSTEM",
    "fraudmlops"
)

ADLS_DIRECTORY = os.getenv(
    "ADLS_DIRECTORY",
    "landing/transactions"
)

BATCH_SIZE = int(
    os.getenv(
        "BATCH_SIZE",
        "1000"
    )
)

DRIFT_MODE = os.getenv(
    "DRIFT_MODE",
    "normal"
).lower()

RANDOM_SEED = os.getenv(
    "RANDOM_SEED"
)

if RANDOM_SEED is not None:
    RANDOM_SEED = int(
        RANDOM_SEED
    )

OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        "output"
    )
)


# ---------------------------------------------------------
# TRANSACTION TYPES
# ---------------------------------------------------------

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
]


# ---------------------------------------------------------
# VALIDATE CONFIGURATION
# ---------------------------------------------------------

def validate_config():

    if not ADLS_ACCOUNT_NAME:
        raise RuntimeError(
            "ADLS_ACCOUNT_NAME is required."
        )

    if not ADLS_FILESYSTEM:
        raise RuntimeError(
            "ADLS_FILESYSTEM is required."
        )

    if not ADLS_DIRECTORY:
        raise RuntimeError(
            "ADLS_DIRECTORY is required."
        )

    if BATCH_SIZE <= 0:
        raise ValueError(
            "BATCH_SIZE must be greater than 0."
        )

    if DRIFT_MODE not in {
        "normal",
        "drift"
    }:
        raise ValueError(
            "DRIFT_MODE must be "
            "'normal' or 'drift'."
        )


# ---------------------------------------------------------
# GENERATE SYNTHETIC TRANSACTIONS
#
# Supporting demo code.
# You do NOT need to study this deeply.
# ---------------------------------------------------------

def generate_transactions(
    batch_size=1000,
    drift_mode="normal",
    seed=None
):

    rng = random.Random(seed)

    # -----------------------------------------------------
    # NORMAL PRODUCTION DISTRIBUTION
    #
    # Calibrated approximately against historical PaySim
    # transaction-type proportions.
    # -----------------------------------------------------

    if drift_mode == "normal":

        type_weights = [
            0.2199,   # CASH_IN
            0.3517,   # CASH_OUT
            0.0065,   # DEBIT
            0.3381,   # PAYMENT
            0.0838    # TRANSFER
        ]

    # -----------------------------------------------------
    # INTENTIONALLY CHANGED DISTRIBUTION
    # Used later for drift testing.
    # -----------------------------------------------------

    else:

        type_weights = [
            0.10,     # CASH_IN
            0.30,     # CASH_OUT
            0.05,     # DEBIT
            0.20,     # PAYMENT
            0.35      # TRANSFER
        ]

    batch_time = datetime.now(
        timezone.utc
    )

    batch_id = str(
        uuid.uuid4()
    )

    transactions = []

    # -----------------------------------------------------
    # GENERATE EACH TRANSACTION
    # -----------------------------------------------------

    for _ in range(batch_size):

        # PaySim step range = 1 to 744.
        # IMPORTANT:
        # Step is generated for every transaction,
        # not once for the entire batch.
        current_step = rng.randint(
            1,
            744
        )

        transaction_type = rng.choices(
            TRANSACTION_TYPES,
            weights=type_weights,
            k=1
        )[0]

        # -------------------------------------------------
        # NORMAL AMOUNT DISTRIBUTION
        #
        # Calibrated approximately against:
        #
        # historical avg amount     ≈ 179,862
        # historical avg log amount ≈ 10.84
        #
        # This is still synthetic data.
        # -------------------------------------------------

        amount = rng.lognormvariate(
            10.84,
            1.59
        )

        # -------------------------------------------------
        # CONTROLLED DRIFT
        #
        # In drift mode:
        # TRANSFER transactions become much larger.
        # -------------------------------------------------

        if (
            drift_mode == "drift"
            and transaction_type == "TRANSFER"
        ):
            amount *= 2.5

        # -------------------------------------------------
        # ACCOUNT BALANCES
        # -------------------------------------------------

        origin_balance = rng.uniform(
            1000,
            500000
        )

        destination_balance = rng.uniform(
            0,
            500000
        )

        new_origin_balance = max(
            origin_balance - amount,
            0
        )

        new_destination_balance = (
            destination_balance
            + amount
        )

        # -------------------------------------------------
        # ACCOUNT IDS
        # -------------------------------------------------

        origin_id = (
            f"C{rng.randint(1000000, 9999999)}"
        )

        destination_id = (
            f"C{rng.randint(1000000, 9999999)}"
        )

        # -------------------------------------------------
        # SYNTHETIC FRAUD PROBABILITY
        #
        # Historical PaySim fraud rate:
        # approximately 0.129%.
        #
        # TRANSFER / CASH_OUT receive a higher risk.
        # Other transaction types receive a very small
        # baseline probability.
        # -------------------------------------------------

        if transaction_type in {
            "TRANSFER",
            "CASH_OUT"
        }:

            fraud_probability = 0.0026

        else:

            fraud_probability = 0.0003

        # -------------------------------------------------
        # INTENTIONAL FRAUD-RATE DRIFT
        # -------------------------------------------------

        if drift_mode == "drift":

            fraud_probability *= 1.5

        fraud_probability = min(
            fraud_probability,
            1.0
        )

        is_fraud = int(
            rng.random()
            < fraud_probability
        )

        # -------------------------------------------------
        # CREATE TRANSACTION RECORD
        # -------------------------------------------------

        transaction = {

            # Unique row ID
            "event_id":
                str(uuid.uuid4()),

            # Identifies which generated batch
            # this transaction belongs to
            "batch_id":
                batch_id,

            "event_time":
                batch_time.isoformat(),

            "schema_version":
                "1.0",

            "step":
                current_step,

            "type":
                transaction_type,

            "amount":
                float(amount),

            "nameOrig":
                origin_id,

            "oldbalanceOrg":
                float(origin_balance),

            "newbalanceOrig":
                float(new_origin_balance),

            "nameDest":
                destination_id,

            "oldbalanceDest":
                float(destination_balance),

            "newbalanceDest":
                float(
                    new_destination_balance
                ),

            "isFraud":
                is_fraud,

            "isFlaggedFraud":
                0,

            # Useful later for drift testing
            "drift_mode":
                drift_mode
        }

        transactions.append(
            transaction
        )

    return transactions


# ---------------------------------------------------------
# SAVE TRANSACTIONS AS LOCAL PARQUET
# ---------------------------------------------------------

def save_parquet(
    transactions
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    filename = (
        f"batch_{timestamp}.parquet"
    )

    output_file = (
        OUTPUT_DIR / filename
    )

    # Python list of dictionaries
    #          ↓
    # PyArrow Table

    table = pa.Table.from_pylist(
        transactions
    )

    # PyArrow Table
    #          ↓
    # Parquet file

    pq.write_table(
        table,
        output_file,
        compression="snappy"
    )

    logger.info(
        "Parquet created: %s",
        output_file
    )

    return output_file


# ---------------------------------------------------------
# CREATE ADLS CLIENT
# ---------------------------------------------------------

def create_adls_client():

    # Locally this can use:
    #
    # az login / Azure CLI credential
    #
    # In Azure it can use other
    # DefaultAzureCredential sources.

    credential = (
        DefaultAzureCredential()
    )

    account_url = (
        f"https://"
        f"{ADLS_ACCOUNT_NAME}"
        f".dfs.core.windows.net"
    )

    service_client = (
        DataLakeServiceClient(
            account_url=account_url,
            credential=credential
        )
    )

    return service_client


# ---------------------------------------------------------
# CREATE ADLS DIRECTORY STRUCTURE
# ---------------------------------------------------------

def get_or_create_directory(
    filesystem_client,
    directory_path
):

    # Example:
    #
    # landing/transactions
    #
    # becomes:
    #
    # landing
    # landing/transactions

    parts = [
        part
        for part
        in directory_path.split("/")
        if part
    ]

    current_path = ""

    directory_client = None

    for part in parts:

        if current_path:

            current_path = (
                f"{current_path}/{part}"
            )

        else:

            current_path = part

        directory_client = (
            filesystem_client
            .get_directory_client(
                current_path
            )
        )

        try:

            directory_client.create_directory()

            logger.info(
                "Created ADLS directory: %s",
                current_path
            )

        except ResourceExistsError:

            # Directory already exists.
            # This is completely fine.
            pass

    return directory_client


# ---------------------------------------------------------
# UPLOAD PARQUET FILE TO ADLS
# ---------------------------------------------------------

def upload_to_adls(
    service_client,
    local_file
):

    # Select filesystem/container:
    #
    # fraudmlopsstorage
    #       ↓
    # fraudmlops

    filesystem_client = (
        service_client
        .get_file_system_client(
            ADLS_FILESYSTEM
        )
    )

    # Get/create:
    #
    # landing/transactions

    directory_client = (
        get_or_create_directory(
            filesystem_client,
            ADLS_DIRECTORY
        )
    )

    # Build remote file reference:
    #
    # landing/transactions/
    # batch_xxx.parquet

    file_client = (
        directory_client
        .get_file_client(
            local_file.name
        )
    )

    # Local Parquet
    #       ↓
    # upload to ADLS

    with open(
        local_file,
        "rb"
    ) as data:

        file_client.upload_data(
            data,
            overwrite=True
        )

    remote_path = (
        f"{ADLS_DIRECTORY}/"
        f"{local_file.name}"
    )

    logger.info(
        "Uploaded to ADLS: %s",
        remote_path
    )

    return remote_path


# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------

def main():

    validate_config()

    logger.info(
        "Starting transaction batch producer"
    )

    logger.info(
        "Storage account: %s",
        ADLS_ACCOUNT_NAME
    )

    logger.info(
        "Filesystem: %s",
        ADLS_FILESYSTEM
    )

    logger.info(
        "ADLS directory: %s",
        ADLS_DIRECTORY
    )

    logger.info(
        "Batch size: %s",
        BATCH_SIZE
    )

    logger.info(
        "Drift mode: %s",
        DRIFT_MODE
    )

    # -----------------------------------------------------
    # STEP 1
    # Generate a new production-like batch
    # -----------------------------------------------------

    transactions = (
        generate_transactions(
            batch_size=BATCH_SIZE,
            drift_mode=DRIFT_MODE,
            seed=RANDOM_SEED
        )
    )

    fraud_count = sum(
        transaction["isFraud"]
        for transaction
        in transactions
    )

    logger.info(
        "Generated transactions: %s",
        len(transactions)
    )

    logger.info(
        "Synthetic fraud rows: %s",
        fraud_count
    )

    # -----------------------------------------------------
    # STEP 2
    # Save as Parquet locally
    # -----------------------------------------------------

    local_file = (
        save_parquet(
            transactions
        )
    )

    # -----------------------------------------------------
    # STEP 3
    # Connect to ADLS
    # -----------------------------------------------------

    service_client = (
        create_adls_client()
    )

    try:

        # -------------------------------------------------
        # STEP 4
        # Upload Parquet batch
        # -------------------------------------------------

        remote_path = (
            upload_to_adls(
                service_client,
                local_file
            )
        )

        logger.info(
            "Batch production completed successfully."
        )

        logger.info(
            "ADLS destination: %s",
            remote_path
        )

    finally:

        service_client.close()

        logger.info(
            "ADLS client closed."
        )


# ---------------------------------------------------------
# APPLICATION ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()