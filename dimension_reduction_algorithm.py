import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class PCA:

    def project(self, Y, l):
        return Y @ self.projections[:,:l]
    
    def __init__(self, X):
        u = X.mean(dim=0)
        Cov = (X-u).T @ (X-u)
        eigenvalues, eigenvectors = torch.linalg.eigh(Cov)
        sort_index = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sort_index]
        eigenvectors = eigenvectors[:, sort_index]
        self.projections = eigenvectors

class LDA:
    
    def project(self, Y, l):
        return Y @ self.pca_projections @ self.projections[:,:l]
    
    def __init__(self, X, labels):
        self.pca_projections, X = PCA_keep(X)
        m = X.shape[0] # sample num
        n = X.shape[1] # dimension
        categories = torch.unique(labels)
        categories_num = len(categories)
        self.categories_num = categories_num 
        u = X.mean(dim=0)
        u_x = torch.zeros(categories_num, n).to(device)
        N_x = torch.zeros(categories_num).to(device)
        for index, value in enumerate(categories):
            mask = labels == value
            N_x[index] = mask.sum().item()
            u_x[index] = X[mask].mean(dim=0)
        N_diag = torch.diag(N_x).to(device)
        S_B = (u_x - u).T @ N_diag @ (u_x - u)
        S_W = torch.zeros(n, n).to(device)
        for i in range(categories_num):
            mask = labels == categories[i]
            S_W += (X[mask] - u_x[i]).T @ (X[mask] - u_x[i])
        A = torch.inverse(S_W) @ S_B
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        eigenvalues, eigenvectors = clear_eigen(eigenvalues, eigenvectors)
        sort_index = torch.argsort(eigenvalues, descending=True)
        sort_index = sort_index[:categories_num - 1] # the constraint
        eigenvalues = eigenvalues[sort_index]
        eigenvectors = eigenvectors[:, sort_index]
        self.projections = eigenvectors
        
def PCA_keep(X, percent=0.98): # reconstruct the data, reserve 98% information 
    u = X.mean(dim=0)
    Cov = (X-u).T @ (X-u)
    eigenvalues, eigenvectors = torch.linalg.eigh(Cov)
    total_variance = torch.sum(eigenvalues) # eigenvalues is in ascending order
    sum_variance = 0
    for i in range(len(eigenvalues)):
        sum_variance += eigenvalues[i]
        if sum_variance > (1 - percent) * total_variance:
            break
    eigenvalues = eigenvalues[i:]
    eigenvectors = eigenvectors[:, i:]
    return eigenvectors, X @ eigenvectors

def clear_eigen(eigenvalues, eigenvectors):
    
    real_indices = torch.abs(eigenvalues.imag) < 1e-2
    # if real_indices.sum() != real_indices.shape[0]:
    #     print(f'Exist {eigenvalues[real_indices.logical_not()].shape[0]} complex eigenvalue', )
    eigenvalues = eigenvalues[real_indices].real
    eigenvectors = eigenvectors[:, real_indices].real
    return eigenvalues, eigenvectors
    
def compute_adjacency_weight(X, k, option='LE', t=0, Delta=0.1, labels=None, kp=None, high_distance=None): # compute the adjancency weight in k nearest neighbors
    m = X.shape[0]
    W = torch.zeros(m,m).to(device)
    if high_distance is None:
        distance = torch.cdist(X, X)
    else:
        distance = high_distance
    k = min(k, m - 1)
    _, indices = torch.topk(distance, (k + 1), largest=False, dim=1)
    indices = indices[:, 1:]
    lm_mark = torch.arange(m).repeat_interleave(k)
    rm_mark = indices.contiguous().view(-1)
    if option == 'LE':
        if t == 0:
            W = torch.full((m,m), float('inf')).to(device)
            W[lm_mark, rm_mark] = distance[lm_mark, rm_mark]
            W = torch.minimum(W, W.T)
            # for i in range(indices.shape[0]):
            #     W[i, indices[i]] = distance[i, indices[i]]
            #     W[indices[i], i] = distance[indices[i], i]
            sig = torch.median(W[W != float('inf')]) * 30
            W = torch.exp( - W / sig)
            return W
        if t < 0:
            W[lm_mark, rm_mark] = 1
            W = torch.maximum(W, W.T)
            return W
        else:
            W = torch.full((m, m), float=('inf')).to(device)
            W[lm_mark, rm_mark] = distance[lm_mark, rm_mark]
            W = torch.minimum(W, W.T)
            W = torch.exp(- W / t)
            return W
    elif option == 'LEE':
        i = 0
        for indice in indices:  # 向量的索引
            subVe = X[i] - X[indice]  # 做差
            Gram = subVe @ subVe.T
            Gram = Gram + torch.trace(Gram) * Delta ** 2 / k * (torch.eye(k).to(device))
            Gram = torch.inverse(Gram)
            W[i, indice] = torch.sum(Gram, dim=1) / torch.sum(Gram)
            i += 1
    else: # option == 'LDE'
        if kp is None:
            kp = k
        kp = min(kp, m - 1)
        _, kp_indices = torch.topk(distance, (kp + 1), largest=False, dim=1)
        kp_indices = kp_indices[:, 1:]
        
        lmp_mark = torch.arange(m).repeat_interleave(kp)
        rmp_mark = kp_indices.contiguous().view(-1)
        Wt = torch.full((m,m), float('inf')).to(device)
        Wtp = torch.full((m,m), float('inf')).to(device)
        Wp = torch.zeros(m,m).to(device)
        
        label_matrix = labels.repeat(m, 1)
        label_matrix = label_matrix == label_matrix.T
        if t == -1:
            Wt[:, :] = 0.
            Wtp[:, :] = 0.
            Wt[lm_mark, rm_mark] = 1
            Wtp[lmp_mark, rmp_mark] = 1
            Wt = torch.maximum(Wt, Wt.T)
            Wtp = torch.maximum(Wp, Wp.T)
        else:
            Wt[lm_mark, rm_mark] = distance[lm_mark, rm_mark]
            Wtp[lmp_mark, rmp_mark] = distance[lmp_mark, rmp_mark]
            Wt = torch.minimum(Wt, Wt.T)
            Wtp = torch.minimum(Wtp, Wtp.T)
            sig = torch.median(Wt[Wt != float('inf')]) * 30
            Wt = torch.exp( - Wt / sig)
            Wtp = torch.exp(- Wtp / sig)
        W[label_matrix] = Wt[label_matrix]
        Wp[label_matrix.logical_not()] = Wtp[label_matrix.logical_not()]
        return W, Wp
    return W
    
def LE(X, N, t=-1):  # 没有限制保留的维数
    # projection_pca, X = PCA_keep(X) # LE don't need the pre-process by PCA
    W = compute_adjacency_weight(X, N, 'LE', t)
    D_diag = torch.sum(W, dim=0)
    D = torch.diag(D_diag)
    L = D - W # Laplacian matrix  Lf = λDf
    M = torch.inverse(D) @ L
    eigenvalues, eigenvectors = torch.linalg.eig(M)
    eigenvalues, eigenvectors = clear_eigen(eigenvalues, eigenvectors)
    
    indices = torch.argsort(eigenvalues)
    indices = indices[1:] # discard the smallest one which correspondings to 0 eigenvalue
    embeddings = eigenvectors[:, indices]    # 现在仅返回按顺序的K-1个特征向量
    # y^TDy = 1, eigenvectors need to be scaled.
    ytdy = torch.sum((embeddings ** 2) * D_diag.unsqueeze(1), dim=0)  # ytdy need to be rescaled to 1
    rescale = torch.sqrt(ytdy)
    embeddings = embeddings / rescale
    return embeddings


class LPP:
    
    def project(self, Y, l):
        return Y @ self.pca_projections @ self.projections[:, :l]
    
    def __init__(self, X, k, t=0, throw_first_component=False):
        self.pca_projections, X = PCA_keep(X)
        W = compute_adjacency_weight(X, k, 'LE', t)
        D_diag = torch.sum(W, dim=0)
        D = torch.diag(D_diag)
        L = D - W
        Lp = X.T @ L @ X
        Dp = X.T @ D @ X
        eigenvalues, eigenvectors = torch.linalg.eig(torch.inverse(Dp) @ Lp)
        eigenvalues, eigenvectors = clear_eigen(eigenvalues, eigenvectors)
        sort_index = torch.argsort(eigenvalues) #  
        eigenvalues = eigenvalues[sort_index]
        eigenvectors = eigenvectors[:, sort_index]

        indices = eigenvalues >= 1e-6
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        if throw_first_component:
            eigenvalues = eigenvalues[1:]
            eigenvectors = eigenvectors[:, 1:]
        self.projections = eigenvectors
   
class kernel_LPP:
    
    def compute_projection(self):
        A = self.K @ self.L @ self.K
        B = self.K @ self.D @ self.K
        C = torch.inverse(B) @ A
        eigenvalues, eigenvectors = torch.linalg.eig(C)
        eigenvalues, eigenvectors = clear_eigen(eigenvalues, eigenvectors)
        sort_index = torch.argsort(eigenvalues) #  
        eigenvalues = eigenvalues[sort_index]
        eigenvectors = eigenvectors[:, sort_index]
        if self.throw_first_component: 
            eigenvalues = eigenvalues[1:]
            eigenvectors = eigenvectors[:, 1:]
        indices = eigenvalues >= 1e-6
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        self.Alpha = eigenvectors
    
    def fit_project(self, testX):
        testX = testX @ self.pca_projections
        distance = torch.cdist(self.X, testX) ** 2
        KB = torch.exp(- distance / (2 * self.gamma ** 2))
        self.KB = KB
    
    def project_l(self, l):
        return self.KB.T @ self.Alpha[:, :l]
    
    def __init__(self, X, k, t=-1, kernel='RBF', gamma=-1, throw_first_component=False):
        self.pca_projections, self.X = PCA_keep(X)
        self.k = k
        self.t = t
        self.kernel = kernel
        self.throw_first_component =  throw_first_component
        m = self.X.shape[0]
        distance = torch.cdist(self.X, self.X) ** 2
        if gamma <= 0:
            distance_mean = distance.mean()
            self.gamma = torch.sqrt(distance_mean).item()
        else:
            self.gamma = gamma
        self.K = torch.exp(-distance / (2 * self.gamma ** 2))
        high_distance = self.K.diag().unsqueeze(1).expand(m, m) + self.K.diag().unsqueeze(0).expand(m, m) - 2 * self.K
        self.W = compute_adjacency_weight(self.X, self.k, 'LE', high_distance=high_distance)
        self.D = torch.diag(self.W.sum(dim=0)).to(device)
        self.L = self.D - self.W
        self.compute_projection()
    
 
class twoD_LPP:    
    def compute_L(self):
        middle_matrix = self.B @ self.R @ self.R.T @ self.B.permute(0, 1, 3, 2)
        RA = self.W.unsqueeze(-1).unsqueeze(-1) * middle_matrix
        RA = RA.sum(dim=(0, 1))
        middle_matrix = self.X @ self.R @ self.R.T @ self.X.permute(0, 2, 1)
        RB = self.D.unsqueeze(-1).unsqueeze(-1) * middle_matrix
        RB = RB.sum(dim=0)
        # for i in range(self.m):
        #     for j in range(self.m):
        #         RA += self.W[i, j] * (self.X[i] - self.X[j]) @ self.R @ self.R.T @ (self.X[i] - self.X[j]).T
        #         RB += self.D[i, j] * self.X[i] @ self.R @ self.R.T @ self.X[j].T
        RC = torch.inverse(RB) @ RA
        eigenvalues, eigenvectors = torch.linalg.eig(RC)
        eigenvalues, eigenvectors = clear_eigen(eigenvalues, eigenvectors)
        indices = eigenvalues >= 1e-6
        eigenvalues[indices]
        eigenvectors[:, indices]
        
        sort_index = torch.argsort(eigenvalues)
        sort_index = sort_index[:self.l1]
        eigenvalues = eigenvalues[sort_index]
        eigenvectors = eigenvectors[:, sort_index]
        if torch.norm(eigenvectors - self.L) < 0.1:    
            return True
        else:
            self.L = eigenvectors
            return False
    
    def compute_R(self):
        middle_matrix = self.B.permute(0, 1, 3, 2) @ self.L @ self.L.T @ self.B
        LA = self.W.unsqueeze(-1).unsqueeze(-1) * middle_matrix
        LA = LA.sum(dim=(0, 1))
        middle_matrix = self.X.permute(0, 2, 1) @ self.L @ self.L.T @ self.X
        LB = self.D.unsqueeze(-1).unsqueeze(-1) * middle_matrix
        LB = LB.sum(dim=0)
        # for i in range(self.m):
        #     for j in range(self.m):
        #         LA += self.W[i, j] * (self.X[i] - self.X[j]).T @ self.L @ self.L.T @ ((self.X[i] - self.X[j]))
        #         LB += self.D[i, j] * self.X[i].T @ self.L @ self.L.T @ self.X[j]
        LC = torch.inverse(LB) @ LA
        eigenvalues, eigenvectors = torch.linalg.eig(LC)
        eigenvalues, eigenvectors = clear_eigen(eigenvalues, eigenvectors)
        indices = eigenvalues >= 1e-6
        eigenvalues[indices]
        eigenvectors[:, indices]
        
        sort_index = torch.argsort(eigenvalues)
        sort_index = sort_index[:self.l2]
        eigenvalues = eigenvalues[sort_index]
        eigenvectors = eigenvectors[:, sort_index]
        if torch.norm(eigenvectors - self.R) < 0.1:    
            return True
        else:
            self.R = eigenvectors
            return False
    
    def __init__(self, X, k, n1, n2, l1, l2, count=30):
        self.k = k
        self.n1 = n1
        self.n2 = n2
        self.l1 = l1
        self.l2 = l2
        self.m = X.shape[0]
        self.X = X.view(self.m, self.n1, self.n2)
        self.L = torch.randn(n1, l1).to(device)
        self.R = torch.randn(n2, l2).to(device)
        self.W = compute_adjacency_weight(X, k, 'LE')
        self.D = self.W.sum(dim=0)
        self.D_diag = torch.diag(self.D).to(device)
        self.B = self.X.unsqueeze(1) - self.X.unsqueeze(0)
        self.count = count
        
        while True:
            if self.compute_L():
                print()
                break
            if self.compute_R():
                print()
                break
            if self.count == 0:
                print()
                break
            print(self.count, end=" | ")
            self.count -= 1
    
    def project(self, Y, l=None):
        Y = Y.view(-1, self.n1, self.n2)
        length = Y.shape[0]
        embeddings = torch.matmul(torch.matmul(self.L.T, Y), self.R)
        return embeddings.view(length, -1)    

def NPE(X, k, throw_zero_component=True, throw_first_component=False): # X, original data points，k nearest neighbors
    projection_pca, X = PCA_keep(X)
    m = X.shape[0] # m samples
    W = compute_adjacency_weight(X, k, 'LLE', Delta=0.1)
    I = torch.eye(m).to(device)
    M = (I - W).T @ (I - W)
    A = X.T @ M @ X
    B = X.T @ X
    C = torch.inverse(B) @ A 
    eigenvalues, eigenvectors = torch.linalg.eig(C)
    eigenvalues, eigenvectors = clear_eigen(eigenvalues, eigenvectors)
    
    sort_index = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_index]
    eigenvectors = eigenvectors[:, sort_index]
    
    if throw_first_component:
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]
    if throw_zero_component:
        indices = eigenvalues >= 1e-6
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
    
    eigenvectors = eigenvectors[:, indices]
    embeddings = X @ eigenvectors
    column_sums_squared = torch.sum(embeddings ** 2, dim=0)  # 归一化,满足 yTy = 1的要求
    column_norms = torch.sqrt(column_sums_squared)
    eigenvectors = eigenvectors / column_norms
    return projection_pca, eigenvectors

def KNN_predict(train_X, train_Y, test_X, k=1):
    distance = torch.cdist(test_X, train_X)  # 距离计算
    _, indices = torch.topk(distance, k, largest=False, dim=1)  # 寻找距离最近的的k个向量
    knn_labels = train_Y[indices]  # 映射为标签
    predicted_labels, _ = torch.mode(knn_labels, dim=1)  # 众数
    return predicted_labels

class LDE:
    def project(self, Y, l):
        return Y @ self.pca_projections @ self.projections[:, :l]
    
    def __init__(self, X, labels, k, t=0, kp=None):
        self.labels = labels
        self.k = k
        self.t = t
        self.pca_projections, X = PCA_keep(X)
        W, Wp = compute_adjacency_weight(X, k, 'LDE', t, labels=self.labels, kp=kp)
        D = W.sum(dim=0); D_diag = torch.diag(D).to(device)
        Dp = Wp.sum(dim=0); Dp_diag = torch.diag(Dp).to(device)
        A = X.T @ (Dp_diag - Wp) @ X
        B = X.T @ (D_diag - W) @ X
        C = torch.inverse(B) @ A
        eigenvalues, eigenvectors = torch.linalg.eig(C)
        eigenvalues, eigenvectors = clear_eigen(eigenvalues, eigenvectors)
        sort_index = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sort_index]
        eigenvectors = eigenvectors[:, sort_index]
        self.projections = eigenvectors

class kernel_LDE:
    
    def fit_project(self, Y):
        distance = torch.cdist(self.X, Y @ self.pca_projections) ** 2
        KB = torch.exp( - distance / (2 * self.gamma ** 2))
        self.KB = KB
        
    def project_l(self, l):
        return self.KB.T @ self.Alpha[:, :l]

    def __init__(self, X, labels, k, t=0, kernel='RBF', gamma=-1, kp=None):
        self.pca_projections, X = PCA_keep(X)
        self.X = X
        self.kernel = 'RBF'
        # self.gamma = X.mean().item()  # choose mean as
        # print(self.gamma)
        distance = torch.cdist(X, X) ** 2
        if gamma <= 0:
            distance_mean = distance.mean()
            self.gamma = torch.sqrt(distance_mean).item()
        else:
            self.gamma = gamma
        K = torch.exp( - distance / (2 * self.gamma ** 2))
        self.K = K
        m = self.X.shape[0]
        high_distance = self.K.diag().unsqueeze(1).expand(m, m) + self.K.diag().unsqueeze(0).expand(m, m) - 2 * self.K
        W, Wp = compute_adjacency_weight(X, k, option='LDE', labels=labels, t=t, kp=kp, high_distance=high_distance)
        D = W.sum(dim=0); D_diag = torch.diag(D).to(device)
        Dp = Wp.sum(dim=0); Dp_diag = torch.diag(Dp).to(device)
        A = K @ (Dp_diag - Wp) @ K
        B = K @ (D_diag - W) @ K
        C = torch.inverse(B) @ A
        eigenvalues, eigenvectors = torch.linalg.eig(C)
        eigenvalues, eigenvectors = clear_eigen(eigenvalues, eigenvectors)
        sort_index = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sort_index]
        eigenvectors = eigenvectors[:, sort_index]
        self.Alpha = eigenvectors
    
def compute_accuracy(predict_labels, test_labels):
    return torch.sum(predict_labels == test_labels).item() / predict_labels.shape[0]
    
def predict_and_compute_acc(predictor, trainX, testX, train_label, test_label, l):
    embedding_train = predictor.project(trainX, l)
    embedding_test = predictor.project(testX, l)
    predict_label = KNN_predict(embedding_train, train_label, embedding_test)
    return compute_accuracy(predict_label, test_label)

class twoD_LDE:
    
    def compute_R(self):
        middle_matrix = self.B.permute(0, 1, 3, 2) @ self.L @ self.L.T @ self.B
        LA = self.Wp.unsqueeze(-1).unsqueeze(-1) * middle_matrix
        LB = self.W.unsqueeze(-1).unsqueeze(-1) * middle_matrix
        LA = LA.sum(dim=(0,1))
        LB = LB.sum(dim=(0,1))
        LC = torch.inverse(LB) @ LA
        eigenvalues, eigenvectors = torch.linalg.eig(LC)
        eigenvalues, eigenvectors = clear_eigen(eigenvalues, eigenvectors)
        sort_index = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sort_index]
        eigenvectors = eigenvectors[:, sort_index]
        eigenvectors = eigenvectors[:, :self.l2]
        if(torch.norm(self.R - eigenvectors) < 0.1):
            return True
        else:
            self.R = eigenvectors
            return False
    
    def compute_L(self):
        middle_matrix = self.B @ self.R @ self.R.T @ self.B.permute(0, 1, 3, 2)
        RA = self.Wp.unsqueeze(-1).unsqueeze(-1) * middle_matrix
        RB = self.W.unsqueeze(-1).unsqueeze(-1) * middle_matrix
        RA = RA.sum(dim=(0,1))
        RB = RB.sum(dim=(0,1))
        RC = torch.inverse(RB) @ RA
        eigenvalues, eigenvectors = torch.linalg.eig(RC)
        eigenvalues, eigenvectors = clear_eigen(eigenvalues, eigenvectors)
        sort_index = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sort_index]
        eigenvectors = eigenvectors[:, sort_index]
        eigenvectors = eigenvectors[:, :self.l1]
        if(torch.norm(self.L - eigenvectors) < 0.1):
            return True
        else:
            self.L = eigenvectors
            return False
        
    
    def __init__(self, X, k, n1, n2, l1, l2, labels, t=-1, kp=None, count=30):
        self.count = count
        self.n1 = n1
        self.n2 = n2
        self.l1 = l1
        self.l2 = l2
        self.m = X.shape[0]
        self.A = X.view(self.m, n1, n2).to(device)
        self.B = self.A.unsqueeze(1) - self.A.unsqueeze(0)
        self.W, self.Wp = compute_adjacency_weight(X, k, 'LDE', labels=labels,kp=kp)
        self.L = torch.randn(n1, l1).to(device)
        self.R = torch.randn(n2, l2).to(device)
        while True:
            if self.compute_L():
                print()
                break
            if self.compute_R():
                print()
                break
            if self.count == 0:
                print()
                break
            print(self.count, end=" | ")
            self.count -= 1
            
            
    def project(self, Y, l=None):
        Y = Y.view(-1, self.n1, self.n2)
        length = Y.shape[0]
        embeddings = torch.matmul(torch.matmul(self.L.T, Y), self.R)
        return embeddings.view(length, -1)
    
def store_code():
    from torchvision import datasets, transforms
    from torch.utils.data import Subset, DataLoader
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)
    train_dataset = Subset(train_dataset, range(2000))
    test_dataset = Subset(test_dataset, range(2000))
    train_data = None
    train_label = None
    test_data = None
    test_label = None
    train_data_loader = DataLoader(train_dataset, batch_size=2000, shuffle=False)
    for data, label in train_data_loader:
        train_data = data.view(2000, -1)
        train_label = label
    test_data_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)
    for data, label in test_data_loader:
        test_data = data.view(2000, -1)
        test_label = label
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    train_label = train_label.to(device)
    test_label = test_label.to(device)
    train_data_num = train_data.shape[0]
    joint_data = torch.cat((train_data, test_data), dim=0)
    labels = torch.cat((train_label, test_label), dim=0)
    return joint_data, labels, train_data_num