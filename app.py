import pandas as pd
import heapq
import math
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def heuristic(a, b):
    # 3D Euclidean distance
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def a_star(nodes, start_node, end_node):
    # nodes: { 'N001': (x, y, z), ... }
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    
    came_from = {}
    g_score = {node: float('inf') for node in nodes}
    g_score[start_node] = 0
    
    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            return path[::-1]

        # In a 3D point cloud, neighbors are often defined by proximity
        # Here we check all nodes; for large datasets, use a KD-Tree
        for neighbor in nodes:
            if neighbor == current: continue
            
            # Example: neighbor is valid if within a certain distance
            dist = heuristic(nodes[current], nodes[neighbor])
            if dist > 50: continue # Adjust '50' based on your coordinate scale
            
            tentative_g_score = g_score[current] + dist
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(nodes[neighbor], nodes[end_node])
                heapq.heappush(open_set, (f_score, neighbor))
                
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    file = request.files['file']
    start = request.form['start']
    end = request.form['end']
    
    df = pd.read_csv(file)
    # Expecting CSV columns: ID, X, Y, Z
    node_data = {row['ID']: (row['X'], row['Y'], row['Z']) for _, row in df.iterrows()}
    
    path = a_star(node_data, start, end)
    return jsonify({"path": path if path else "No path found"})

if __name__ == '__main__':
    app.run(debug=True)