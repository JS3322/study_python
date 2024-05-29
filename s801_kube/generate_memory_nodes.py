import subprocess
import json
import yaml

def get_all_nodes():
    result = subprocess.run(["kubectl", "get", "nodes", "-o", "json"], capture_output=True, text=True)
    nodes = json.loads(result.stdout)['items']
    return [node['metadata']['name'] for node in nodes]

def get_node_memory_usage(node_name):
    result = subprocess.run(["kubectl", "top", "node", node_name, "-o", "json"], capture_output=True, text=True)
    node_info = json.loads(result.stdout)
    memory_usage = node_info['usage']['memory']
    return memory_usage

def get_high_memory_nodes(threshold):
    node_names = get_all_nodes()
    high_memory_nodes = []

    for node_name in node_names:
        memory_usage = get_node_memory_usage(node_name)
        # Convert memory usage from Mi to Ki for comparison
        memory_usage_k = int(memory_usage.strip('Mi')) * 1024
        if memory_usage_k <= threshold:
            high_memory_nodes.append(node_name)
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