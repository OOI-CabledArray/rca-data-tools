import os

def select_logger():
    from prefect import get_run_logger
    try:
        logger = get_run_logger()
    except:
        from loguru import logger
    
    return logger


def coerce_qartod_executed_to_int(ds):
    logger = select_logger()

    qartod_executed_vars = [var for var in ds.variables if 'qartod_executed' in var]
    for var in qartod_executed_vars:
        executed_tests = ds[var].tests_executed.replace(' ', '').split(',')
    
        for i, test in enumerate(executed_tests):
            test_var_name = f"{var}_{test}"
            ds[test_var_name] = ds[var].str[i].astype(int)

        ds = ds.drop(var)
    logger.info(f"ds size post coercion: {ds.nbytes}")
    return ds


def get_s3_kwargs():
    aws_key = os.environ.get("AWS_KEY")
    aws_secret = os.environ.get("AWS_SECRET")
    
    s3_kwargs = {'key': aws_key, 'secret': aws_secret}
    return s3_kwargs