import os

input_file = "sleepdrive.py"
output_file = "runnable_sleepdrive.py"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Fix Colab display function
    if "display(" in line:
        line = line.replace("display(", "print(")
        
    # Fix Google Drive paths
    if "/content/drive/MyDrive/Sleep Diso/MODELS/" in line:
        line = line.replace("/content/drive/MyDrive/Sleep Diso/MODELS/", "model/")
    if "/content/drive/MyDrive/Sleep Diso/METRICS/" in line:
        line = line.replace("/content/drive/MyDrive/Sleep Diso/METRICS/", "metrics/")
    if "/content/drive/" in line:
        line = line.replace("/content/drive/", "./")
        
    # Prevent matplotlib from blocking the script execution
    if "plt.show()" in line:
        line = line.replace("plt.show()", "# plt.show()")
        
    new_lines.append(line)

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Successfully cleaned {input_file} and saved to {output_file}")
