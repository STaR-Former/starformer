from typing import List

__all__ = [
    'DatasetOptions',
    'TrainingMethodOptions',
    'DatasetPartitionOptions',
    'MaskingOptions',
    'ModelOptions'
]

class ModelOptions:
    lstm: str = 'lstm'
    starformer: str = 'starformer'
    all: list[str] = [
        'lstm',
        'starformer'
    ]

class DatasetOptions:
    mnist: str = 'mnist'
    geolife: str = 'geolife'
    # ucr_uea
    ethanolconcentration: str='ethanolconcentration'
    eigenworms: str='eigenworms'
    facedetection: str='facedetection'
    heartbeat: str='heartbeat'
    handwriting: str='handwriting'
    insectwingbeat: str='insectwingbeat'
    japanesevowels: str='japanesevowels'
    pendigits: str='pendigits'
    pemssf: str='pemssf'
    rightwhalecalls: str='rightwhalecalls'
    selfregulationscp1: str='selfregulationscp1'
    selfregulationscp2: str='selfregulationscp2'
    spokenarabicdigits: str='spokenarabicdigits'
    uwavegesturelibrary: str='uwavegesturelibrary'
    pam: str='pam'

    ucr_uea: List[str]=[
        'ethanolconcentration', 'eigenworms', 'facedetection', 'heartbeat',
        'handwriting', 'insectwingbeat', 'japanesevowels', 'pendigits',
        'pemssf', 'rightwhalecalls', 'selfregulationscp1', 'selfregulationscp2',
        'spokenarabicdigits', 'uwavegesturelibrary']
    
    ucr_uea_binary: List[str]=['facedetection', 'heartbeat', 'rightwhalecalls', 'selfregulationscp1', 'selfregulationscp2']

    ucr_uea_multi: List[str]=[
        'ethanolconcentration', 'eigenworms', 'handwriting', 'insectwingbeat', 
        'japanesevowels', 'pendigits', 'pemssf', 'spokenarabicdigits', 'uwavegesturelibrary']
    

class TrainingMethodOptions:
    centralized: str = 'centralized'
    federated: str = 'federated'


class DatasetPartitionOptions:
    # i.i.d
    iid: str = 'iid'  
    # non i.i.d
    shard: str = 'shard' 
    dirichlet: str = 'dirichlet'


class MaskingOptions:
    drm: str = 'drm'
    random: str = 'random'


