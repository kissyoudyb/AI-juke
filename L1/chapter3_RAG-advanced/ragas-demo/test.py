import os
import langchain_community

# 获取 langchain_community 的安装目录
community_path = os.path.dirname(langchain_community.__file__)

# 递归查找包含 ParentDocumentRetriever 的文件
def find_class_file(folder, class_name):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if class_name in content:
                            # 转换为导入路径格式
                            rel_path = os.path.relpath(file_path, community_path)
                            import_path = 'langchain_community.' + rel_path[:-3].replace(os.sep, '.')
                            print(f"找到 {class_name} 在: {import_path}")
                            return import_path
                except:
                    continue
    return None

# 执行查找
class_path = find_class_file(community_path, 'ParentDocumentRetriever')
if not class_path:
    print("未找到该类，建议降级 langchain_community 版本！")