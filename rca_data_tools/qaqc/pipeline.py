"""pipeline.py

This module contains the qaqc_pipeline entry point: main() and the QAQCPipeline
class. This class interfaces with the cli entry point to orchestrate a pipeline 
with prefect 2 which uses the zarr files in the ooi-data s3 bucket to generate 
plots as pngs. These plots are viewable throug the frontend web app in QAQC_dashboard.

Prefect 2 @flow and @task decorated functions are found in flow.py

"""
import datetime
import argparse
import time
from loguru import logger

from prefect.deployments import run_deployment

from rca_data_tools.qaqc.constants import (
    instrument_dict,
    sites_dict,
    stage2_dict,
    stage3_dict,
    SPAN_DICT,
)
from rca_data_tools.qaqc.constants import COMPUTE_EXCEPTIONS, CAM_SPANS, THROTTLE_SPANS
from rca_data_tools.qaqc.flow import qaqc_pipeline_flow, S3_BUCKET

now = datetime.datetime.utcnow()
all_configs_dict = {**sites_dict, **stage2_dict, **stage3_dict}

class QAQCPipeline:
    """
    QAQC Pipeline Class to create Pipeline for specified site, time, and span.

    """
    def __init__(
        self,
        site=None,
        time=now.strftime("%Y-%m-%d"),
        span='1',
        threshold=5_000_000,
        cloud_run=True,
        s3_bucket=S3_BUCKET,
        s3_sync=True,
        s3fs_kwargs={},
    ):
        self.site = site
        self.time = time
        self.span = span
        self.threshold = threshold
        self.stage = self._lookup_stage()
        self._cloud_run = cloud_run
        self.s3_bucket = s3_bucket
        self.s3_sync = s3_sync
        self.s3fs_kwargs = s3fs_kwargs
        self.valid_spans = SPAN_DICT
        self._site_ds = {}
        self._params_valid = True

        self.__setup()

    def __setup(self):
        # TODO data filtering/verification should occur in this class
        self.created_dt = datetime.datetime.utcnow()
        if self.site not in all_configs_dict:
            raise ValueError(
                f"{self.site} is not available. Available sites {','.join(list(all_configs_dict.keys()))}"  # noqa
            )
        self._site_ds = all_configs_dict[self.site]

        self.plotInstrument = self._site_ds.get('instrument', None)
        if 'CAM' in self.site:
            self.valid_spans = CAM_SPANS
        if 'SPKIRA' in self.site or 'OPTAA' in self.site:
            self.valid_spans = THROTTLE_SPANS #FIXME remove once we can reduce runtimes on these

        if self.span not in self.valid_spans:
            logger.warning(
                f"span {self.span} not valid. Must be {','.join(list(self.valid_spans.keys()))}"  # noqa
            )
            self._params_valid = False

        self.name = f"{self.site}--{self.span}"
    
    def _lookup_stage(self):
        # instance needs to be aware of what stage its instrument is
        if self.site in sites_dict.keys():
            return 1
        elif self.site in stage2_dict.keys():
            return 2
        elif self.site in stage3_dict.keys():
            return 3

    def __repr__(self):
        return f"<{self.name}>"
    
    @property
    def cloud_run(self):
        return self._cloud_run

    @cloud_run.setter
    def cloud_run(self, cr):
        self._cloud_run = cr

    @property
    def parameters(self):
        """
        OOI plot parameters
        """
        return (
            instrument_dict[self.plotInstrument]['plotParameters']
            .replace('"', '')
            .split(',')
        )

    @property
    def flow_parameters(self):
        """
        Prefect flow parameters
        """
        return {
            'site': self.site,
            'timeString': self.time,
            'span': self.span,
            'threshold': self.threshold,
            'stage': self.stage,
            'fs_kwargs': self.s3fs_kwargs,
            'sync_to_s3': self.s3_sync,
            's3_bucket': self.s3_bucket,
        }


    def run(self, parameters=None):
        """
        Runs the flow either in the cloud or locally.
        """
        if self._params_valid == False:
            logger.info(f"{self.name} with span {self.span} is not a valid combination skipping...")
        else:
            if parameters is None:
                parameters = self.flow_parameters
        
            logger.info(f"parameters set to: {parameters}!")
            if self.cloud_run is True:
                run_name = "-".join([str(self.site), str(self.span)])
                # IMPORTANT run_deployment determines the infrastructure and resources for each flow_run
                if self.site in COMPUTE_EXCEPTIONS and self.span in COMPUTE_EXCEPTIONS[self.site]:

                    deployment_name = f"qaqc-pipeline-flow/{COMPUTE_EXCEPTIONS[self.site][self.span]}"
                    logger.warning(f"{self.site} with span {self.span} requires additional compute resources, creating flow_run from {deployment_name} instead of default")
                    run_deployment(
                        name=deployment_name,
                        parameters=parameters,
                        flow_run_name=run_name,
                        timeout=10
                    )
                # otherwise run the default deployment with default compute resources        
                else:
                    run_deployment(
                        name="qaqc-pipeline-flow/2vcpu_16gb",
                        parameters=parameters,
                        flow_run_name=run_name,
                        timeout=10 # timeout might need to be increase if we have race condition errors
                    )
            else:
                qaqc_pipeline_flow(**parameters)


def run_stage(stage_dict, args):
    for key in stage_dict.keys():
        logger.info(f"creating pipeline instance for site: {key}")
        pipeline = QAQCPipeline(
            site=key,
            time=args.time,
            span=args.span,
            threshold=args.threshold,
            cloud_run=args.cloud,
            s3_bucket=args.s3_bucket,
            s3_sync=args.s3_sync,
        )
        if args.run is True:
            pipeline.run()
        # Add 20s delay for each run #TODO is this really necessary? 
        time.sleep(20)


def parse_args():
    arg_parser = argparse.ArgumentParser(description='QAQC Pipeline cli')

    arg_parser.add_argument('--stage1', action="store_true", help="run all stage1 instruments")
    arg_parser.add_argument('--stage2', action='store_true', help="run all stage2 instruments")
    arg_parser.add_argument('--stage3', action='store_true', help="run all stage3 instruments")
    arg_parser.add_argument('--run', action="store_true")
    arg_parser.add_argument('--cloud', action="store_true")
    arg_parser.add_argument('--s3-sync', action="store_true")
    arg_parser.add_argument('--site', type=str, default=None)
    arg_parser.add_argument('--time', type=str, default=now.strftime("%Y-%m-%d"))
    arg_parser.add_argument(
        '--s3-bucket',
        type=str,
        default=S3_BUCKET,
        help="S3 Bucket to store the plots.",
    )
    arg_parser.add_argument(
        '--span',
        type=str,
        default='7',
        help=f"Choices {str(list(SPAN_DICT.keys()))}",
    )
    arg_parser.add_argument('--threshold', type=int, default=5000000)

    return arg_parser.parse_args()


def main():
    args = parse_args()

    # arg validation
    if args.site and any([args.stage1, args.stage2, args.stage3]):
        raise ValueError("Do not use `--site` and `--stage` arguments together. "
            "Run either individual sites OR stage groups of instruments.")

    if args.stage1 is True:
        run_stage(sites_dict, args)
    if args.stage2 is True:
        run_stage(stage2_dict, args)
    if args.stage3 is True:
        run_stage(stage3_dict, args)

    if args.site is not None:
        # Creates only one pipeline instance, useful for testing
        pipeline = QAQCPipeline(
            site=args.site,
            time=args.time,
            span=args.span,
            threshold=args.threshold,
            cloud_run=args.cloud,
            s3_bucket=args.s3_bucket,
            s3_sync=args.s3_sync,
        )

        if args.run is True:
            pipeline.run()

if __name__ == '__main__':
    main()