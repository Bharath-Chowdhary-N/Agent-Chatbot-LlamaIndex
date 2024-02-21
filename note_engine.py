from llama_index.core.tools import FunctionTool
import os

note_file = os.path.join("data","notes.txt")

def save_note(note):
    if not os.path.isfile(note_file):
        open(note_file,"w")
    with open(note_file,"a") as f:
        f.writelines([note + "\n"])
    
    return "Congrats! note is saved!"

note_engine = FunctionTool.from_defaults(fn=save_note, name="note_saver", description="this tool helps to save txt note")