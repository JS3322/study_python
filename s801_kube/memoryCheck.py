import subprocess
import json
import os

def get_high_memory_nodes(threshold):
    result = subprocess.run(["kubectl", "get", "nodes", "-o", "json"], capture_output=True, text=True)
    nodes = json.loads(result.stdout)['items']
    high_memory_nodes = []

    for node in nodes:
        for condition in node['status']['conditions']:
            if condition['type'] == 'Ready' and condition['status'] == 'True':
                mem_capacity = node['status']['capacity']['memory']
                if int(mem_capacity.strip('Ki')) >= threshold:
                    high_memory_nodes.append(node['metadata']['name'])
    return high_memory_nodes

def generate_values_file(high_memory_nodes):
    values_content = {
        "memoryCondition": high_memory_nodes
    }
    with open('values_dynamic.yaml', 'w') as f:
        yaml.dump(values_content, f)

if __name__ == '__main__':
    threshold_memory = 8000000  # 8GB in Ki
    high_memory_nodes = get_high_memory_nodes(threshold_memory)
    generate_values_file(high_memory_nodes)