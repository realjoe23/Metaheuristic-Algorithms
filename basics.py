#%% Bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt

#%% Esqueleto de función objetivo
class Objective_function:
    def __init__(self, dim, lim_inf, lim_sup):
        self.d = dim
        self.lb = lim_inf
        self.ub = lim_sup
    
    def eval(self, X):
        return np.array(list(map(self.function,list(X.T)))) 
        
    def function(self, x):
        pass
    
    def draw2d(self, resolution=200, name=None):
        '''
        Esta función dibuja la gráfica de contorno de
        una función que recibe dos parámetros flotantes y entrega uno.
        
        Parameters
        ----------
        funcion : f:R^2->R
            función que recibe dos valores flotantes y regresa un valor flotante
        lim_inf : np.array de dos flotantes
            limite inferior del espacio de dibujo, por ejemplo: (-5, -5)
        lim_sup : np.array de dos flotantes
            limite inferior del espacio de dibujo, por ejemplo: (5, 5)

        Returns
        -------
        Figura
        '''
        # Revisar si se tiene la dimensión adecuada
        if self.d != 2:
            raise('No podemos dibujar la función ya que no tiene dos dimensiones')
                
        # Crear puntos a dibujar en el dominio de la funcion
        x = np.linspace(self.lb[0], self.ub[0], resolution)
        y = np.linspace(self.lb[1], self.ub[1], resolution)
        xx, yy = np.meshgrid(x, y)
        
        # Evaluar los puntos en la funcion 
        data = (np.c_[xx.ravel(), yy.ravel()]).T
        z = self.eval(data)
        zz = z.reshape(xx.shape)
        
        # Dibujar
        plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm,)
        plt.xlabel(r'$x$', fontsize=16)
        plt.ylabel(r'$y$', fontsize=16)
        plt.xlim(self.lb[0], self.ub[0])
        plt.ylim(self.lb[1], self.ub[1])   
        

        if name is not None:
            plt.title(name, fontsize=20)
                  
        
    def draw3d(self, resolution=200, name=None):  
        '''
        Esta función dibuja la superficie 3D de una función que recibe
        dos parámetros flotantes y regresa uno.
        
        Parameters
        ----------
        funcion : f:R^2->R
            función que recibe dos valores flotantes y regresa un valor flotante
        lim_inf : tupla de dos flotantes
            limite inferior del espacio de dibujo, por ejemplo: (-5, -5)
        lim_sup : tupla de dos flotantes
            limite inferior del espacio de dibujo, por ejemplo: (5, 5)

        Returns
        -------
        Figure.
        '''
        
        # Revisar si se tiene la dimensión adecuada
        if self.d != 2:
            raise('No podemos dibujar la función ya que no tiene dos dimensiones')
            
        # Crear puntos a dibujar en el dominio de la funcion
        x = np.linspace(self.lb[0], self.ub[0], resolution)
        y = np.linspace(self.lb[1], self.ub[1], resolution)
        xx, yy = np.meshgrid(x, y)
        
        # Evaluar los puntos en la funcion 
        data = (np.c_[xx.ravel(), yy.ravel()]).T
        z = self.eval(data)
        zz = z.reshape(xx.shape)
        
        # Dibujar
        fig = plt.gcf()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_surface(xx, yy, zz, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.5)
        ax.set_xlabel(r'$x$', fontsize=16)
        ax.set_ylabel(r'$y$', fontsize=16)
        ax.set_zlabel(r'$f(x,y)$', fontsize=16)
        ax.set_xlim(self.lb[0], self.ub[0])
        ax.set_ylim(self.lb[1], self.ub[1])    
        
        if name is not None:
            plt.title(name, fontsize=20) 
    
    def parallel_coordinates(self, X):
        evaluation = np.array(list(map(self.function, list(X.T))))
        norm = plt.Normalize(vmin=min(evaluation), vmax=max(evaluation))
        cmap = plt.get_cmap('coolwarm')
        
        fig = plt.gcf()
        fig.figsize = (15,5)
        gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1,4])
        ax1 = fig.add_subplot(gs[0,0])
        
        ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax1.xaxis.set_label_position('top')
        ax1.axhline(y=0, color='k', linestyle='-')
        ax1.scatter(evaluation, np.zeros((X.shape[1])), marker='o', s=70,
                        c=cmap(norm(evaluation)))
        ax1.set_xlabel('Evaluation', fontsize=14)
        ax1.set_yticks([])
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        
        ax2 = fig.add_subplot(gs[1,0])
        dimensions = np.arange(1, X.shape[0]+1)
        for i in range(1, X.shape[0]+1):
            ax2.axvline(x=i, color='k', linestyle='-')
        for i in range(X.shape[1]):
            ax2.plot(dimensions, X[:,i],
                     color=cmap(norm(evaluation[i])))
            
            
        ax2.set_xticks(dimensions)  
        ax2.set_xlabel('Dimension', fontsize=14)
        ax2.set_ylabel('Value', fontsize=14)
        ax2.set_ylim([min(self.lb), max(self.ub)])
        fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1,4])
        
class Animation():
    def __init__(self, animation_type='2d', name=None):
        self.type = animation_type
        self.fig = None
        self.pts = None
        self.best = None
        self.name = name
        
    def initialize_animation(self, ObjFunc, x, ig):  
        if self.type == '2d': 
            self.fig = plt.figure(figsize=(8,8))
            ObjFunc.draw2d(resolution=300, name=self.name)
            self.pts = plt.scatter(x[0,:], x[1,:], c='b')
            self.best = plt.scatter(x[0, ig], x[1, ig], c='r')
        elif self.type == '3d':
            self.fig = plt.figure(figsize=(8,8))
            ObjFunc.draw3d(resolution=300, name=self.name)
            self.pts = plt.plot(x[0,:], x[1,:], ObjFunc.eval(x), '.b')
            self.best = plt.plot(x[0,ig], x[1,ig], ObjFunc.function(x[:,ig]), '.r')
        elif self.type == 'nd':
            self.fig = plt.figure(figsize=(15,5))
            self.pts = ObjFunc.parallel_coordinates(x)

    def update_animation(self, ObjFunc, x, ig):
        plt.pause(0.5)
        if self.type == '2d':   
            self.pts.remove()
            self.best.remove()
            self.pts = plt.scatter(x[0,:], x[1,:], c='b')
            self.best = plt.scatter(x[0, ig], x[1, ig], c='r')
        elif self.type == '3d':
            self.pts[0].remove()
            self.best[0].remove()
            self.pts=plt.plot(x[0,:], x[1,:], ObjFunc.eval(x), '.b', linewidth=15)
            self.best=plt.plot(x[0,ig], x[1,ig], ObjFunc.function(x[:,ig]), '.r', linewidth=15)
        elif self.type == 'nd':
            self.fig.clear()
            self.pts = ObjFunc.parallel_coordinates(x)