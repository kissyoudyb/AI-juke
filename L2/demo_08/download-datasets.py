#数据集下载
# pip install datasets oss2 addict "datasets<3.0.0"
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('w10442005/ruozhiba_qa', subset_name='default', split='train',cache_dir="/root/datasets")
#您可按需配置 subset_name、split，参照“快速使用”示例代码