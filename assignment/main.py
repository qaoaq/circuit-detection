from process_circuit_image import process_circuit_image
from analyze_circuit_connectivity import analyze_circuit_connectivity


image_path = "circuit2.jpg"
final_image, detected_components = process_circuit_image(image_path)
analyze_circuit_connectivity(image_path, final_image, detected_components)