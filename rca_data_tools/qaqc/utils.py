import os
import numpy as np
import fsspec

def select_logger():
    from prefect import get_run_logger
    try:
        logger = get_run_logger()
    except:
        print('Could not start prefect logger...running local log')
        from loguru import logger
    
    return logger


def coerce_qartod_executed_to_int(ds):
    logger = select_logger()

    logger.info(f"ds size pre coercion: {ds.nbytes}")
    qartod_executed_vars = [var for var in ds.variables if 'qartod_executed' in var]
    for var in qartod_executed_vars:
        executed_tests = ds[var].tests_executed.replace(' ', '').split(',')
    
        for i, test in enumerate(executed_tests):
            test_var_name = f"{var}_{test}"
            ds[test_var_name] = ds[var].str[i].astype(int)

        ds = ds.drop(var)
    logger.info(f"ds size post coercion: {ds.nbytes}")
    return ds


def prepare_s3_bucket(bucket_name):
    logger = select_logger()

    # Load secrets from gh runner because this function is called outside of prefect flow
    aws_key = os.environ.get('AWS_KEY')
    aws_secret = os.environ.get('AWS_SECRET')
    fs_kwargs = {'key': aws_key, 'secret': aws_secret}

    S3FS = fsspec.filesystem('s3', **fs_kwargs)
    logger.info("Collecting existing 'profile' image files.")

    existing_files = S3FS.ls(f's3://{bucket_name}/')
    files_to_delete = [file for file in existing_files if 'profile' in file]
    logger.info(f"To be deleted: {files_to_delete} ")
    # The number of profiles changes so we want to delete old profile files.
    for f in files_to_delete:
        S3FS.rm(f)

    logger.info("'Profile' files deleted.")
