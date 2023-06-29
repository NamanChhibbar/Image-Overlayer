import numpy as np, cv2
from numpy.linalg import norm

class K_means:
    """
    Implementation of K-means clustering algorithm.
    """

    data: np.ndarray
    k: int
    centroids: list[np.ndarray]
    clusters: list[list[np.ndarray]]
    cost: int

    def __init__(self, data: np.ndarray, k: int, seed: int=-1):
        if seed >= 0:
            np.random.seed(seed)
        self.data = data
        self.k = k

        # Choosing random centroids
        indices = np.random.choice(np.arange(len(self.data)), k)
        self.centroids = list(self.data[indices])

        # Running a single step of K-means
        self._update_clusters()
        self._update_centroids()
        self._update_cost()

    def _update_clusters(self) -> list[list[np.ndarray]]:
        """
        Updates clusters based on current centroids.

        Returns
            clusters: List of updated clusters.
        """
        clusters = []
        for i in range(self.k): clusters.append([])
        centroids = self.centroids
        for data in self.data:
            min_dist = norm(data - centroids[0])
            min_dist_index = 0
            for i in range(1, self.k):
                dist = norm(data - centroids[i])
                if dist < min_dist:
                    min_dist = dist
                    min_dist_index = i
            clusters[min_dist_index].append(data)

        self.clusters = clusters
        return clusters
    
    def _update_centroids(self) -> list[np.ndarray]:
        """
        Updates centroids by taking mean of the respective cluster.

        Returns
            centroids: List of updated centroids.
        """
        centroids = self.centroids
        clusters = self.clusters
        empty_clusters = []

        for i in range(len(clusters)):
            if len(clusters[i]) == 0:
                empty_clusters.append(i)
                continue
            mean = np.zeros(centroids[0].shape)
            for point in clusters[i]:
                mean += point
            mean /= len(clusters[i])
            centroids[i] = mean
        
        empty_clusters.reverse()
        for i in empty_clusters:
            clusters.pop(i)
            centroids.pop(i)
        self.k -= len(empty_clusters)
        self.clusters = clusters
        self.centroids = centroids
        return centroids
    
    def _update_cost(self) -> int:
        """
        Updates the cost.

        Returns
            cost: Updated cost.
        """
        centroids = self.centroids
        clusters = self.clusters
        cost = 0
        for centroid, cluster in zip(centroids, clusters):
            for point in cluster:
                cost += norm(point - centroid)
        self.cost = cost / len(self.data)
        return self.cost

    def run(self, threshold: float) -> list[list[np.ndarray]]:
        """
        Runs the K-means algorithm until cost > threshold.

        Parameters
            threshold: Specifies when to stop K-means.

        Returns
            clusters: List of clusters at the end of K-means.
        """
        prev_cost = self._update_cost()
        while True:
            self._update_clusters()
            self._update_centroids()
            curr_cost = self._update_cost()
            if prev_cost - curr_cost < threshold:
                break
            else:
                prev_cost = curr_cost
        return self.clusters

class Edge:
    """
    Class to generate and handle edges in an image.
    """
    
    image: np.ndarray
    edges: np.ndarray
    grid: np.ndarray

    def __init__(self, im):
        self.image = im
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        self.edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    def create_grid(self, n: int, threshold: float) -> np.ndarray:
        """
        Creates an n x n grid in edges image. A cell has value 1 if number of white pixels in it are above the given threshold.

        Parameters
            n: Size of the grid.
            threshold: Number of white pixels in a cell above which cell has value 1.

        Retruns
            grid: 2D numpy array containing cell values.
        """
        edges = self.edges
        h, w = edges.shape
        pixels_per_cell = np.ceil((h * w) / (n * n))
        i_step = np.ceil(h / n).astype(int)
        j_step = np.ceil(w / n).astype(int)
        grid = []
        for i in range(n):
            row = []
            for j in range(n):
                cell = edges[i*i_step: min((i+1)*i_step, h), j*j_step: min((j+1)*j_step, w)]
                white_pixels = np.sum(cell == 255)
                row.append(1 if (white_pixels < threshold * pixels_per_cell) else 0)
            grid.append(row)
        self.grid = np.array(grid)
        return self.grid
    
    def rectangle_pos(self, rec_dim: tuple|list, error: float) -> np.ndarray:
        """
        Finds the optimal positions for a rectangle with given dimensions based on number of cells with value 1.

        Parameters
            rec_dim: Dimensions of the rectangle in the format (length, breadth).
            error: Maximum fraction of cells with 0 value in rectangle.

        Returns
            pos: 2D numpy array containing optimal positions of rectangle.
        """
        b, l = self.edges.shape
        grid = self.grid
        n = grid.shape[0]
        cell_l = l // n
        cell_b = b // n
        rec_rows = np.ceil(rec_dim[1] / cell_b).astype(int)
        rec_cols = np.ceil(rec_dim[0] / cell_l).astype(int)
        num_cells = rec_rows * rec_cols
        pos = []
        if (rec_cols > n):
            raise ValueError("Length of rectangle is too large for length of image")
        if (rec_rows > n):
            raise ValueError("Breadth of rectangle is too large for breadth of image")
        for i in range(0, n - rec_rows + 1, 2):
            for j in range(0, n - rec_cols + 1, 2):
                rec = grid[i: i+rec_rows, j: j+rec_cols]
                on_cells = np.sum(rec)
                if (on_cells >= (1 - error) * num_cells):
                    pos.append([j * cell_l, i * cell_b])
        return np.array(pos)

def create_mask(image: np.ndarray, im_dim: tuple, mask_choice: str, mask_dim: np.ndarray, brightness: float):
    """
    Creates a mask over the given image.

    Parameters
        im_dim: Dimensions of the image in the format (length, breadth).
        mask_choice: Choice of mask. Must be one of whole, top, bottom, left, or right.
        brightness: Specifies the brightness of the mask. Must be between 0 and 1.

    Returns
        image: Image with mask.
    """
    match mask_choice:
        case "whole":
            image = cv2.convertScaleAbs(image, 1, brightness)
        case "left":
            portion = image[:, :mask_dim[0]]
            image[:, :mask_dim[0]] = cv2.convertScaleAbs(portion, 1, brightness)
        case "right":
            portion = image[:, im_dim[0] - mask_dim[0]:]
            image[:, im_dim[0] - mask_dim[0]:] = cv2.convertScaleAbs(portion, 1, brightness)
        case "top":
            portion = image[:mask_dim[1], :]
            image[:mask_dim[1], :] = cv2.convertScaleAbs(portion, 1, brightness)
        case "bottom":
            portion = image[im_dim[1] - mask_dim[1]:, :]
            image[im_dim[1] - mask_dim[1]:, :] = cv2.convertScaleAbs(portion, 1, brightness)
    return image

def process_text(text: str, size_choice: str, im_dim: tuple|list, word_lim: int, char_lim: int):
    """
    Creates lines based on word and characters limits and calculates font size based on size_choice and image length.

    Parameters
        text: Text to process.
        size_choice: Size of text to overlay. Must be one of small, medium, or large.
        im_dim: Dimensions of image in the format (length, breadth)
        word_lim: Maximum number of words in a line.
        char_lim: Maximum number of characters in a line.

    Returns
        lines: List of strings denoting each line.
        font_size: Absolute value of font size.
    """
    words = text.split()
    lines = []
    line = " "
    line_len = 0
    num_words = 0
    for word in words:
        if num_words == word_lim or line_len + len(word) > char_lim:
            lines.append(line.strip())
            line = word
            line_len = len(word)
            num_words = 1
        else:
            line += f" {word}"
            line_len += len(word)
            num_words += 1
    lines.append(line.strip())

    match size_choice:
        case "small":
            font_size = int(4.4e-2 * im_dim[0])
        case "medium":
            font_size = int(5.4e-2 * im_dim[0])
        case "large":
            font_size = int(6.4e-2 * im_dim[0])
    return lines, font_size

def calc_pos(lines: list[str], padding: int, writer, text_font):
    """
    Calculates position of lines relative to upper left corner of rectangle.

    Parameters
        lines: List of lines.
        padding: Padding between the lines.
        writer: PIL writer object.
        text_font: PIL font object.

    Returns
        2D numpy array containing position of each line in a row.
    """
    rec_dim = [0, (len(lines) - 1) * padding]
    x_pos = []
    y_pos = [0]
    for line in lines:
        dim = writer.textbbox((0, 0), line, text_font)[-2:]
        rec_dim[0] = max(rec_dim[0], dim[0])
        rec_dim[1] += dim[1]
        x_pos.append(dim[0])
        y_pos.append(y_pos[-1] + dim[1] + padding)
    x_pos = (rec_dim[0] - np.array(x_pos)) // 2
    y_pos = np.array(y_pos[: -1])
    return np.array([x_pos, y_pos]).T, rec_dim

def process_image(image: np.ndarray, strip_frac: float, blur_frac: float, blur: int):
    """
    Pre-processes image by stripping off the sides and blurring the center for the edges finding algorithm.

    Parameters
        image: Image to be processed.
        strip_frac: Fraction of image dimensions to be stripped.
        blur_frac: Fraction of image dimensions to be blurred.
        blur: Intensity of blur.
    
    Returns
        Processed image.
    """
    b, l = image.shape[:2]
    strip_l = int(l * strip_frac / 2)
    strip_b = int(b * strip_frac / 2)
    if (blur_frac > 0):
        blur_l = int(l * blur_frac / 2)
        blur_b = int(b * blur_frac / 2)
        blur_cut = image[b//2-blur_b: b//2+blur_b, l//2-blur_l: l//2+blur_l]
        image[b//2-blur_b: b//2+blur_b, l//2-blur_l: l//2+blur_l] = cv2.blur(blur_cut, (blur, blur))
    return image[strip_b: b-strip_b, strip_l: l-strip_l], strip_l, strip_b

def best_spot(image: np.ndarray, n: int, rec_dim: tuple[int, int], num_clst: int, seed: int, im_num):
    """
    Finds the best spot to place the rectangle.

    Parameters
        image: The image.
        n: Size of grid to create in edges image.
        rec_dim: Dimensions of rectangle in the format (length, breadth).
        num_clust: Number of clusters to create using K-means.
        seed: Seed to use in K-means.
    
    Returns
        best_pos: Best spot for the rectangle.
    """
    edge_inst = Edge(image)
    cv2.imwrite(f"/Users/naman/Desktop/Examples/edges/edges{im_num}.jpg", edge_inst.edges)

    th = 0
    while True:
        edge_inst.create_grid(n, th)
        if (np.any(edge_inst.grid)):
            break
        th += 0.001
    e = 0
    while True:
        positions = edge_inst.rectangle_pos(rec_dim, e)
        if (positions.size > 0):
            break
        e += 0.04
    print(f"e: {e}")
    print(f"th: {th}")
    print(f"num of positions: {positions.shape[0]}")
    
    k_inst = K_means(positions.astype(float), num_clst, seed)
    k_inst.run(0.05)
    positions = np.array(k_inst.centroids, dtype=int)

    best_pos = None
    min_white = -1
    for pos in positions:
        x, y = pos
        rect = edge_inst.edges[y: y + rec_dim[1], x: x + rec_dim[0]]
        white_pix = np.sum(rect == 255)
        if (white_pix < min_white or min_white == -1):
            best_pos = pos
            min_white = white_pix
    return best_pos
