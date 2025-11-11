from ultralytics import YOLO
import cv2
import chess
import gradio as gr
import zipfile
import os

if not os.path.exists("model/model.pt"):
    with zipfile.ZipFile("model.zip", "r") as zip_ref:
        zip_ref.extractall("model/")

# Export model
model = YOLO("model/model.pt")

def parse_fen(image, castling_rights, enpassant_sq, turn):

    # Inference called on image to create bounding box
    results = model.predict(source=image, conf=0.4, save=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names
    img = results[0].orig_img

    board_box = None
    # Get box and class for each sub-image
    for i in range(len(boxes)):
        box = boxes[i]
        cls = classes[i]

        # If class is board assign board_box
        if names[int(cls)] == "board":
            board_box = box
            break

    # If no board detected return image and error
    if board_box is None:
        return img, "No board detected!"

    # Board/piece cell creation
    x1, y1, x2, y2 = board_box
    board_w = x2 - x1
    board_h = y2 - y1
    cell_w = board_w / 8
    cell_h = board_h / 8

    # Map pieces to squares
    positions = {} # Dict containing piece and square
    
    # Loop through sub-images to get box and class
    for i in range(len(boxes)):
        box = boxes[i]
        cls = classes[i]
        label = names[int(cls)]
        
        # Ignore board
        if label == "board":
            continue
        
        # Piece center
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        
        # Convert piece coords to board coords
        col = int((cx - x1) / cell_w)
        row = int((cy - y1) / cell_h)
        col = max(0, min(7, col))
        row = max(0, min(7, row))
        
        # Identify file and rank
        file = "abcdefgh"[col]
        rank = str(8 - row)
        
        # Build square from file and rank
        square = file + rank
        positions[square] = label

    # Create chess board from detected pieces using python-chess
    board = chess.Board(None)
    for square, piece_type in positions.items():
        sq = chess.parse_square(square)
        piece = chess.Piece.from_symbol(piece_type)
        board.set_piece_at(sq, piece)
    
    # Add castling rights to FEN
    if len(castling_rights) == 0:
        board.set_castling_fen("-")
    else:
        rights = ""
        for right in castling_rights:
            if right == "WhiteKingside":
                rights += "K"
            elif right == "WhiteQueenside":
                rights += "Q"
            elif right == "BlackKingside":
                rights += "k"
            elif right == "BlackQueenside":
                rights += "q"
        board.set_castling_fen(rights)
    
    # Add enpassant square to FEN
    if enpassant_sq is not None and enpassant_sq is not "":
        board.ep_square = chess.parse_square(enpassant_sq)

    # Set turn to whose move it is
    board.turn = turn
    
    # Python-chess internally handles invalid board states, 
    # such as being able to castle with a different king or rook position,
    # or having a non-legal enpassant square 
    
    # Get FEN from board
    fen = board.fen()
    
    # Draw grid and labels
    img_copy = img.copy()
    for i in range(9):
        cv2.line(img_copy, (int(x1), int(y1 + i * cell_h)), (int(x2), int(y1 + i * cell_h)), (255, 255, 255), 1)
        cv2.line(img_copy, (int(x1 + i * cell_w), int(y1)), (int(x1 + i * cell_w), int(y2)), (255, 255, 255), 1)
    for square, label in positions.items():
        col = "abcdefgh".index(square[0])
        row = 8 - int(square[1])
        cx = int(x1 + col * cell_w + cell_w / 2)
        cy = int(y1 + row * cell_h + cell_h / 2)
        cv2.putText(img_copy, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

    # Convert from CV2 BGR to gradio RGB
    return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB), fen

# Gradio application
demo = gr.Interface(
    fn = parse_fen,
    inputs = [gr.Image(type="filepath", label="Upload a Chessboard Photo"),
              gr.CheckboxGroup(
                choices=["WhiteKingside", "WhiteQueenside", "BlackKingside", "BlackQueenside"],
                value=["WhiteKingside", "WhiteQueenside", "BlackKingside", "BlackQueenside"],
                label="Castling Rights"),
                gr.Textbox(label="Enpassant Square"),
                gr.Checkbox(label="White to Move?")

            ],
    outputs = [
        gr.Image(type="numpy", label="Detected Pieces & Grid"),
        gr.Textbox(label="Generated FEN Position")
    ],
    title="Chess Piece Detector & FEN Generator",
    description="Upload a chessboard image and select options. Generates the FEN representation.",
    examples=[],
    flagging_mode="never"
)

demo.launch(debug=True)
