import numpy as np
import matplotlib.pyplot as plt

# K-means алгоритмын класс
class KMeansFromScratch:
    def __init__(self, n_clusters=2, max_iters=300, tol=1e-4, random_state=None):
        """
        K-means кластерлалын алгоритм
        
        Parameters:
        -----------
        n_clusters : int, default=2
            Кластеруудын тоо
        max_iters : int, default=300
            Хамгийн их давталтын тоо
        tol : float, default=1e-4
            Төвүүдийн өөрчлөлтийн зөвшөөрөгдөх хязгаар
        random_state : int, default=None
            Санамсаргүй тоо үүсгэгчийн төлөв
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
    
    def fit(self, X):
        """
        Өгөгдөлд K-means моделийг тохируулах
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Сургалтын өгөгдөл
        
        Returns:
        --------
        self : object
            Тохируулсан кластерын объект
        """
        # Санамсаргүй төлөвийг тохируулах
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Өгөгдлийн хэмжээг авах
        n_samples, n_features = X.shape
        
        # Анхны төвүүдийг санамсаргүйгээр сонгох
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        # Өмнөх төвүүдийг хадгалах
        prev_centroids = np.zeros_like(self.centroids)
        
        # Кластерын хуваарилалтыг хадгалах
        self.labels_ = np.zeros(n_samples)
        
        # Төвүүд тогтворжтол давтах
        for i in range(self.max_iters):
            # Өгөгдөл бүрийг хамгийн ойр төвтэй кластерт хуваарилах
            self.labels_ = self._assign_clusters(X)
            
            # Өмнөх төвүүдийг хадгалах
            prev_centroids = np.copy(self.centroids)
            
            # Кластер бүрийн төвийг шинэчлэх
            for j in range(self.n_clusters):
                if np.sum(self.labels_ == j) > 0:  # Хоосон кластер үүсэхээс сэргийлэх
                    self.centroids[j] = np.mean(X[self.labels_ == j], axis=0)
            
            # Төвүүдийн өөрчлөлтийг шалгах
            if np.sum((self.centroids - prev_centroids) ** 2) < self.tol:
                break
        
        return self
    
    def _assign_clusters(self, X):
        """
        Өгөгдөл бүрийг хамгийн ойр төвтэй кластерт хуваарилах
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Өгөгдөл
        
        Returns:
        --------
        labels : array, shape (n_samples,)
            Кластерын хуваарилалт
        """
        # Өгөгдөл бүр болон төв бүрийн хоорондын зайг тооцоолох
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            # Евклидийн зайг тооцоолох
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        
        # Хамгийн ойр төвтэй кластерыг сонгох
        return np.argmin(distances, axis=1)
    
    def predict(self, X):
        """
        Шинэ өгөгдлийн кластерыг таамаглах
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Шинэ өгөгдөл
        
        Returns:
        --------
        labels : array, shape (n_samples,)
            Таамагласан кластерууд
        """
        return self._assign_clusters(X)


# Өөрийн гараар өгөгдөл үүсгэх функц
def generate_clustered_data(n_samples=1000, n_features=2, n_clusters=2, cluster_std=1.0, random_state=None):
    """
    Кластертай өгөгдөл үүсгэх
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Үүсгэх өгөгдлийн тоо
    n_features : int, default=2
        Шинжүүдийн тоо
    n_clusters : int, default=2
        Кластеруудын тоо
    cluster_std : float, default=1.0
        Кластерын стандарт хазайлт
    random_state : int, default=None
        Санамсаргүй тоо үүсгэгчийн төлөв
    
    Returns:
    --------
    X : array, shape (n_samples, n_features)
        Үүсгэсэн өгөгдөл
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Кластер бүрийн төвийг үүсгэх
    centers = np.random.randn(n_clusters, n_features) * 5
    
    # Кластер бүрийн өгөгдлийн тоог тооцоолох
    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples % n_clusters
    
    # Өгөгдөл хадгалах массив
    X = np.zeros((n_samples, n_features))
    
    # Кластер бүрийн өгөгдлийг үүсгэх
    start_idx = 0
    for i in range(n_clusters):
        # Тухайн кластерын өгөгдлийн тоо
        n_cluster_samples = samples_per_cluster + (1 if i < remainder else 0)
        end_idx = start_idx + n_cluster_samples
        
        # Кластерын төвийн эргэн тойронд санамсаргүй өгөгдөл үүсгэх
        X[start_idx:end_idx] = centers[i] + np.random.randn(n_cluster_samples, n_features) * cluster_std
        
        start_idx = end_idx
    
    # Өгөгдлийг холих
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    return X[indices]


# Өгөгдөл үүсгэх
X = generate_clustered_data(n_samples=1000, n_features=2, n_clusters=2, random_state=4)

# Өөрийн бичсэн K-means моделийг тодорхойлох
model = KMeansFromScratch(n_clusters=2, random_state=42)

# Моделд тохируулах
model.fit(X)

# Жишээ бүрт кластерын оноох
yhat = model.predict(X)

# Ялгаатай кластеруудыг хайх
clusters = np.unique(yhat)

# Кластер бүрийн дээжүүдээр тархалтын график үүсгэх
plt.figure(figsize=(10, 6))
for cluster in clusters:
    # Тухайн кластерын дээжийн энгэний индексийг авах
    row_ix = np.where(yhat == cluster)
    # Дээжүүдээр тархалтын график зурах
    plt.scatter(X[row_ix, 0], X[row_ix, 1], label=f'Кластер {cluster}')

# Төвүүдийг тэмдэглэх
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], 
            s=300, c='red', marker='*', label='Төвүүд')

plt.title('K-means кластерлал (Бүрэн өөрийн бичсэн алгоритм)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()