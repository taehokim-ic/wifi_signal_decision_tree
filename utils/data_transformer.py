from numpy import loadtxt, ndarray


class DataTransformer:
    @classmethod
    def transform_dataset_to_ndarray(cls, file_path: str) -> ndarray:
        return loadtxt(file_path)
