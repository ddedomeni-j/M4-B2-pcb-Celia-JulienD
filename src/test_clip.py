from option_c_clip import *

import time

CLASS_PROMPTS_1: dict[str, str] = {
    "ok": "a photograph of a clean PCB with no defects",
    "copper":"An image of a defective PCB; copper has leaked onto the circuit.",
    "pin_hole":"An image of a defective PCB; one of the three solder joints is incomplete.",
    "mousebite":"An image of a defective PCB; one of the circuit straight line is broken.",
    "spur":"An image of a defective PCB; a triangle spur is set on top of the circuit.",
    "short":"An image of a defective PCB; one or more of the straight lines are shorter than others.",
    "open":"An image of a defective PCB; one of the straight lines is broken",
}

CLASS_PROMPTS_2: dict[str, str] = {
    "ok": "a close-up image of a printed circuit board with regular traces, clean geometry, symmetric patterns and no visible defect. Image has 3 perfect vertical lines, 3 perfect horizontal lines and 3 plain disks with no holes",
    "copper":"a photo of a printed circuit board with a copper defect. The image has n ellipse on top of the correct circuit",
    "pin_hole":"a printed circuit board with a missing hole defect. One of the 3 disk or line is cropped by a disk",
    "mousebite":"a printed circuit board with a mouse bite defect. One of the straight line is cropped",
    "spur":"a printed circuit board with a spur defect. This image contains a triangle",
    "short":"a printed circuit board with a short circuit defect. There is an asymetric line on the image cutting horizontal and vertical line",
    "open":"a printed circuit board with an open circuit defect. 3 vertical lines, 3 horizontal lines and at least one of the lines has a hole",
}

CLASS_PROMPTS_3: dict[str, str] = {
    "ok": "a close-up image of a printed circuit board with regular traces, clean geometry, symmetric patterns and no visible defect. Image has 3 perfect vertical lines, 3 perfect horizontal lines and 3 plain disks with no holes",
    "ko": "a photo of a defectuous printed circuit. Defect can be an ellipse on top of the correct circuit, a missing hole on one of the disks, a line broken or shorter than the others. An asymetric line is also a defect",
}


processor, model = load_clip_model()

results = evaluate_zero_shot(
    Path("data/test"),
    processor,
    model,
    CLASS_PROMPTS_2
)

print(results)