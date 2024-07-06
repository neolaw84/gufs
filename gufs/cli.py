import logging 
from pathlib import Path 
from glob import glob

from tqdm.auto import tqdm
import cv2

from gufs.preprocess.mask import mask_by_query, Query, dilate_mask, remove_by_mask_using_deep_image_prior

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Cli:
    def generate_mask_by_query(self, input_dir:str, output_dir:str, query_dir:str, query_mask_dir:str=None):
        if query_mask_dir is None: query_mask_dir = Path(query_dir) / "masks"
        input_files = glob(f"{input_dir}/*.png") + glob(f"{input_dir}/*.jpg") + glob(f"{input_dir}/*.png")
        query_files = glob(f"{query_dir}/*.png") + glob(f"{query_dir}/*.jpg") + glob(f"{query_dir}/*.png")
        query_files_stems = [Path(qf).name for qf in query_files]

        def _get_query_mask_file_path(qfs):
            qm = Path(query_mask_dir) / qfs 
            if not qm.exists():
                logger.warn(f"query mask {qm} not found")
                return None 
            else:
                return str(qm)
            
        def _create_query(qf, qm):
            query_image = cv2.imread(qf, cv2.IMREAD_GRAYSCALE)
            query_mask = cv2.imread(qm, cv2.IMREAD_GRAYSCALE)
            return Query(query_image, query_mask)

        query_mask_files = [_get_query_mask_file_path(qfs) for qfs in tqdm(query_files_stems)]
        queries = [_create_query(qf, qm) for qf, qm in tqdm(zip(query_files, query_mask_files)) if qm]

        logger.info("start processing ... ")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for input_f in tqdm(input_files):
            input_image = cv2.imread(input_f, cv2.IMREAD_GRAYSCALE)
            mask = mask_by_query(queries, input_image)
            cv2.imwrite(str(Path(output_dir) / Path(input_f).name), mask)

    def dilate_mask(self, input_dir:str, output_dir:str, dilation_factor:int=1, inv:bool=False):
        input_files = glob(f"{input_dir}/*.png") + glob(f"{input_dir}/*.jpg") + glob(f"{input_dir}/*.png")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for input_f in tqdm(input_files):
            input_image = cv2.imread(input_f, cv2.IMREAD_GRAYSCALE)
            dilated = dilate_mask(input_image, dilation_factor=dilation_factor, inv=inv)
            cv2.imwrite(f"{Path(output_dir) / Path(input_f).name}", dilated)

    def remove_watermark_by_mask(self, input_dir:str, input_mask_dir:str, output_dir:str, inv_mask:bool=False, max_dim:int=1920, reg_noise=0.03, input_depth=32, learning_rate=0.01, steps=150):
        input_files = glob(f"{input_dir}/*.png") + glob(f"{input_dir}/*.jpg") + glob(f"{input_dir}/*.png")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for input_f in tqdm(input_files):
            try:
                input_mask_f = str(Path(input_mask_dir) / Path(input_f).name)
                #print (input_mask_f, input_f)
                output_f= str(Path(output_dir) / Path(input_f).name)
                remove_by_mask_using_deep_image_prior(
                    input_f, input_mask_f, output_f, 
                    inv_mask=inv_mask, max_dim=max_dim, reg_noise=reg_noise, 
                    input_depth=input_depth, learning_rate=learning_rate, training_steps=steps
                )
            except Exception as e:
                import traceback
                print(e)
                traceback.print_tb(e.__traceback__) 
                logger.warn(f"problem with {input_f}")
