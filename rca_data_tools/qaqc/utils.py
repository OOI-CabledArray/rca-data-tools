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


def prepare_s3_bucket(bucket_name, fs_kwargs={}):
    logger = select_logger()

    S3FS = fsspec.filesystem('s3', **fs_kwargs)
    logger.info("collecting existing 'profile' image files")

    existing_files = S3FS.ls(f's3://{bucket_name}/')
    files_to_delete = [file for file in existing_files if 'profile' in file]
    # The number of profiles changes so we want to delete old profile files.
    for f in files_to_delete:
        S3FS.rm(f)

    logger.info("'profile' files deleted")
