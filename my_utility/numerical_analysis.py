import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from bisect import bisect_right
import matplotlib as mpl
from scipy import sparse
from sklearn import base
from sklearn import preprocessing
from sklearn import utils

class FAMD_utility:
    def compute_svd(X, n_components, n_iter, random_state, engine):
        """Computes an SVD with k components."""
        from sklearn.utils import extmath
        try:
            import fbpca
            FBPCA_INSTALLED = True
        except ImportError:
            FBPCA_INSTALLED = False
        # Determine what SVD engine to use
        if engine == 'auto':
            engine = 'sklearn'

        # Compute the SVD
        if engine == 'fbpca':
            if FBPCA_INSTALLED:
                U, s, V = fbpca.pca(X, k=n_components, n_iter=n_iter)
            else:
                raise ValueError('fbpca is not installed; please install it if you want to use it')
        elif engine == 'sklearn':
            U, s, V = extmath.randomized_svd(
                X,
                n_components=n_components,
                n_iter=n_iter,
                random_state=random_state
            )
        else:
            raise ValueError("engine has to be one of ('auto', 'fbpca', 'sklearn')")

        U, V = extmath.svd_flip(U, V)

        return U, s, V

    def stylize_axis(ax, grid=True):
        import numpy as np
        from scipy import linalg
        from collections import OrderedDict
        GRAY = OrderedDict([
            ('light', '#bababa'),
            ('dark', '#404040')
        ])
        if grid:
            ax.grid()

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.axhline(y=0, linestyle='-', linewidth=1.2, color=GRAY['dark'], alpha=0.6)
        ax.axvline(x=0, linestyle='-', linewidth=1.2, color=GRAY['dark'], alpha=0.6)

        return ax


    def build_ellipse(X, Y):
        """Construct ellipse coordinates from two arrays of numbers.
        Args:
            X (1D array_like)
            Y (1D array_like)
        Returns:
            float: The mean of `X`.
            float: The mean of `Y`.
            float: The width of the ellipse.
            float: The height of the ellipse.
            float: The angle of orientation of the ellipse.
        """
        import numpy as np
        from collections import OrderedDict
        from scipy import linalg
        
        x_mean = np.mean(X)
        y_mean = np.mean(Y)

        cov_matrix = np.cov(np.vstack((X, Y)))
        U, s, V = linalg.svd(cov_matrix, full_matrices=False)

        chi_95 = np.sqrt(4.61)  # 90% quantile of the chi-square distribution
        width = np.sqrt(cov_matrix[0][0]) * chi_95 * 2
        height = np.sqrt(cov_matrix[1][1]) * chi_95 * 2

        eigenvector = V.T[0]
        angle = np.arctan(eigenvector[1] / eigenvector[0])

        return x_mean, y_mean, width, height, angle

    def make_labels_and_names(X):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            row_label = X.index.name if X.index.name else 'Rows'
            row_names = X.index.tolist()
            col_label = X.columns.name if X.columns.name else 'Columns'
            col_names = X.columns.tolist()
        else:
            row_label = 'Rows'
            row_names = list(range(X.shape[0]))
            col_label = 'Columns'
            col_names = list(range(X.shape[1]))

        return row_label, row_names, col_label, col_names



class CA(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, n_components=2, n_iter=10, copy=True, check_input=True, benzecri=False,
                 random_state=None, engine='auto'):
        self.n_components = n_components
        self.n_iter = n_iter
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.benzecri = benzecri
        self.engine = engine

    def fit(self, X, y=None):

        # Check input
        if self.check_input:
            utils.check_array(X)

        # Check all values are positive
        if (X < 0).any().any():
            raise ValueError("All values in X should be positive")

        _, row_names, _, col_names = FAMD_utility.make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if self.copy:
            X = np.copy(X)

        # Compute the correspondence matrix which contains the relative frequencies
        X = X / np.sum(X)

        # Compute row and column masses
        self.row_masses_ = pd.Series(X.sum(axis=1), index=row_names)
        self.col_masses_ = pd.Series(X.sum(axis=0), index=col_names)

        # Compute standardised residuals
        r = self.row_masses_.to_numpy()
        c = self.col_masses_.to_numpy()
        S = sparse.diags(r ** -.5) @ (X - np.outer(r, c)) @ sparse.diags(c ** -.5)

        # Compute SVD on the standardised residuals
        self.U_, self.s_, self.V_ = FAMD_utility.compute_svd(
            X=S,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine
        )

        # Compute total inertia
        self.total_inertia_ = np.einsum('ij,ji->', S, S.T)

        return self

    def transform(self, X):
        """Computes the row principal coordinates of a dataset.
        Same as calling `row_coordinates`. In most cases you should be using the same
        dataset as you did when calling the `fit` method. You might however also want to included
        supplementary data.
        """
        utils.validation.check_is_fitted(self)
        if self.check_input:
            utils.check_array(X)
        return self.row_coordinates(X)
    @property
    def GRAY(self):
        from collections import OrderedDict
        return OrderedDict([
            ('light', '#bababa'),
            ('dark', '#404040')
        ])  
    @property
    def eigenvalues_(self):
        """The eigenvalues associated with each principal component.
        Benzecri correction is applied if specified.
        """
        utils.validation.check_is_fitted(self)

        K = len(self.col_masses_)

        if self.benzecri:
            return [
                (K / (K - 1.) * (s - 1. / K)) ** 2
                if s > 1. / K else 0
                for s in np.square(self.s_)
            ]

        return np.square(self.s_).tolist()

    @property
    def explained_inertia_(self):
        """The percentage of explained inertia per principal component."""
        utils.validation.check_is_fitted(self)
        return [eig / self.total_inertia_ for eig in self.eigenvalues_]

    def row_coordinates(self, X):
        """The row principal coordinates."""
        utils.validation.check_is_fitted(self)

        _, row_names, _, _ = FAMD_utility.make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            try:
                X = X.sparse.to_coo().astype(float)
            except AttributeError:
                X = X.to_numpy()

        if self.copy:
            X = X.copy()

        # Normalise the rows so that they sum up to 1
        if isinstance(X, np.ndarray):
            X = X / X.sum(axis=1)[:, None]
        else:
            X = X / X.sum(axis=1)

        return pd.DataFrame(
            data=X @ sparse.diags(self.col_masses_.to_numpy() ** -0.5) @ self.V_.T,
            index=row_names
        )

    def column_coordinates(self, X):
        """The column principal coordinates."""
        utils.validation.check_is_fitted(self)

        _, _, _, col_names = FAMD_utility.make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            is_sparse = X.dtypes.apply(pd.api.types.is_sparse).all()
            if is_sparse:
                X = X.sparse.to_coo()
            else:
                X = X.to_numpy()

        if self.copy:
            X = X.copy()

        # Transpose and make sure the rows sum up to 1
        if isinstance(X, np.ndarray):
            X = X.T / X.T.sum(axis=1)[:, None]
        else:
            X = X.T / X.T.sum(axis=1)

        return pd.DataFrame(
            data=X @ sparse.diags(self.row_masses_.to_numpy() ** -0.5) @ self.U_,
            index=col_names
        )

    def plot_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                                   show_row_labels=True, show_col_labels=True, **kwargs):
        """Plot the principal coordinates."""

        utils.validation.check_is_fitted(self)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add style
        ax = FAMD_utility.stylize_axis(ax)

        # Get labels and names
        row_label, row_names, col_label, col_names = FAMD_utility.make_labels_and_names(X)

        # Plot row principal coordinates
        row_coords = self.row_coordinates(X)
        ax.scatter(
            row_coords[x_component],
            row_coords[y_component],
            **kwargs,
            label=row_label
        )

        # Plot column principal coordinates
        col_coords = self.column_coordinates(X)
        ax.scatter(
            col_coords[x_component],
            col_coords[y_component],
            **kwargs,
            label=col_label
        )

        # Add row labels
        if show_row_labels:
            x = row_coords[x_component]
            y = row_coords[y_component]
            for xi, yi, label in zip(x, y, row_names):
                ax.annotate(label, (xi, yi))

        # Add column labels
        if show_col_labels:
            x = col_coords[x_component]
            y = col_coords[y_component]
            for xi, yi, label in zip(x, y, col_names):
                ax.annotate(label, (xi, yi))

        # Legend
        ax.legend()

        # Text
        ax.set_title('Principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

        return ax



    
class PCA(base.BaseEstimator, base.TransformerMixin):
    """Principal Component Analysis (PCA).
    Parameters:
        rescale_with_mean (bool): Whether to substract each column's mean or not.
        rescale_with_std (bool): Whether to divide each column by it's standard deviation or not.
        n_components (int): The number of principal components to compute.
        n_iter (int): The number of iterations used for computing the SVD.
        copy (bool): Whether to perform the computations inplace or not.
        check_input (bool): Whether to check the consistency of the inputs or not.
        as_array (bool): Whether to output an ``numpy.ndarray`` instead of a ``pandas.DataFrame``
            in ``tranform`` and ``inverse_transform``.
    """

    def __init__(self, rescale_with_mean=True, rescale_with_std=True, n_components=2, n_iter=3,
                 copy=True, check_input=True, random_state=None, engine='auto', as_array=False):
        self.n_components = n_components
        self.n_iter = n_iter
        self.rescale_with_mean = rescale_with_mean
        self.rescale_with_std = rescale_with_std
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.engine = engine
        self.as_array = as_array

    def fit(self, X, y=None):

        # Check input
        if self.check_input:
            utils.check_array(X)

        # Convert pandas DataFrame to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)

        # Copy data
        if self.copy:
            X = np.array(X, copy=True)

        # Scale data
        if self.rescale_with_mean or self.rescale_with_std:
            self.scaler_ = preprocessing.StandardScaler(
                copy=False,
                with_mean=self.rescale_with_mean,
                with_std=self.rescale_with_std
            ).fit(X)
            X = self.scaler_.transform(X)

        # Compute SVD
        self.U_, self.s_, self.V_ = FAMD_utility.compute_svd(
            X=X,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine
        )

        # Compute total inertia
        self.total_inertia_ = np.sum(np.square(X))

        return self

    def transform(self, X):
        """Computes the row principal coordinates of a dataset.
        Same as calling `row_coordinates`. In most cases you should be using the same
        dataset as you did when calling the `fit` method. You might however also want to included
        supplementary data.
        """
        utils.validation.check_is_fitted(self)
        if self.check_input:
            utils.check_array(X)
        rc = self.row_coordinates(X)
        if self.as_array:
            return rc.to_numpy()
        return rc

    def inverse_transform(self, X):
        """Transforms row projections back to their original space.
        In other words, return a dataset whose transform would be X.
        """
        utils.validation.check_is_fitted(self)
        X_inv = np.dot(X, self.V_)

        if hasattr(self, 'scaler_'):
            X_inv = self.scaler_.inverse_transform(X_inv)

        if self.as_array:
            return X_inv

        # Extract index
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(data=X_inv, index=index)

    def row_coordinates(self, X):
        """Returns the row principal coordinates.
        The row principal coordinates are obtained by projecting `X` on the right eigenvectors.
        """
        utils.validation.check_is_fitted(self)

        # Extract index
        index = X.index if isinstance(X, pd.DataFrame) else None

        # Copy data
        if self.copy:
            X = np.array(X, copy=True)

        # Scale data
        if hasattr(self, 'scaler_'):
            X = self.scaler_.transform(X)

        return pd.DataFrame(data=X.dot(self.V_.T), index=index, dtype=np.float64)

    def row_standard_coordinates(self, X):
        """Returns the row standard coordinates.
        The row standard coordinates are obtained by dividing each row principal coordinate by it's
        associated eigenvalue.
        """
        utils.validation.check_is_fitted(self)
        return self.row_coordinates(X).div(self.eigenvalues_, axis='columns')

    def row_contributions(self, X):
        """Returns the row contributions towards each principal component.
        Each row contribution towards each principal component is equivalent to the amount of
        inertia it contributes. This is calculated by dividing the squared row coordinates by the
        eigenvalue associated to each principal component.
        """
        utils.validation.check_is_fitted(self)
        return np.square(self.row_coordinates(X)).div(self.eigenvalues_, axis='columns')

    def row_cosine_similarities(self, X):
        """Returns the cosine similarities between the rows and their principal components.
        The row cosine similarities are obtained by calculating the cosine of the angle shaped by
        the row principal coordinates and the row principal components. This is calculated by
        squaring each row projection coordinate and dividing each squared coordinate by the sum of
        the squared coordinates, which results in a ratio comprised between 0 and 1 representing the
        squared cosine.
        """
        utils.validation.check_is_fitted(self)
        squared_coordinates = np.square(self.row_coordinates(X))
        total_squares = squared_coordinates.sum(axis='columns')
        return squared_coordinates.div(total_squares, axis='rows')

    def column_correlations(self, X):
        """Returns the column correlations with each principal component."""
        utils.validation.check_is_fitted(self)

        # Convert numpy array to pandas DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        row_pc = self.row_coordinates(X)

        return pd.DataFrame({
            component: {
                feature: row_pc[component].corr(X[feature])
                for feature in X.columns
            }
            for component in row_pc.columns
        }).sort_index()

    @property
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        utils.validation.check_is_fitted(self)
        return np.square(self.s_).tolist()

    @property
    def explained_inertia_(self):
        """Returns the percentage of explained inertia per principal component."""
        utils.validation.check_is_fitted(self)
        return [eig / self.total_inertia_ for eig in self.eigenvalues_]

    def plot_row_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                             labels=None, color_labels=None, ellipse_outline=False,
                             ellipse_fill=True, show_points=True, **kwargs):
        """Plot the row principal coordinates."""
        utils.validation.check_is_fitted(self)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add style
        ax = FAMD_utility.stylize_axis(ax)

        # Make sure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Retrieve principal coordinates
        coordinates = self.row_coordinates(X)
        x = coordinates[x_component].astype(np.float)
        y = coordinates[y_component].astype(np.float)

        # Plot
        if color_labels is None:
            ax.scatter(x, y, **kwargs)
        else:
            for color_label in sorted(list(set(color_labels))):
                mask = np.array(color_labels) == color_label
                color = ax._get_lines.get_next_color()
                # Plot points
                if show_points:
                    ax.scatter(x[mask], y[mask], color=color, **kwargs, label=color_label)
                # Plot ellipse
                if (ellipse_outline or ellipse_fill):
                    x_mean, y_mean, width, height, angle = FAMD_utility.build_ellipse(x[mask], y[mask])
                    ax.add_patch(mpl.patches.Ellipse(
                        (x_mean, y_mean),
                        width,
                        height,
                        angle=angle,
                        linewidth=2 if ellipse_outline else 0,
                        color=color,
                        fill=ellipse_fill,
                        alpha=0.2 + (0.3 if not show_points else 0) if ellipse_fill else 1
                    ))

        # Add labels
        if labels is not None:
            for xi, yi, label in zip(x, y, labels):
                ax.annotate(label, (xi, yi))

        # Legend
        ax.legend()

        # Text
        ax.set_title('Row principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

        return ax    

    
    
class MCA(CA):

    def fit(self, X, y=None):

        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_initial_columns = X.shape[1]

        # One-hot encode the data
        one_hot = pd.get_dummies(X)

        # Apply CA to the indicator matrix
        super().fit(one_hot)

        # Compute the total inertia
        n_new_columns = one_hot.shape[1]
        self.total_inertia_ = (n_new_columns - n_initial_columns) / n_initial_columns

        return self

    def row_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return super().row_coordinates(pd.get_dummies(X))

    def column_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return super().column_coordinates(pd.get_dummies(X))

    def transform(self, X):
        """Computes the row principal coordinates of a dataset."""
        utils.validation.check_is_fitted(self)
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])
        return self.row_coordinates(X)

    def plot_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                         show_row_points=True, row_points_size=10,
                         row_points_alpha=0.6, show_row_labels=False,
                         show_column_points=True, column_points_size=30, show_column_labels=False,
                         legend_n_cols=1):
        """Plot row and column principal coordinates.
        Parameters:
            ax (matplotlib.Axis): A fresh one will be created and returned if not provided.
            figsize ((float, float)): The desired figure size if `ax` is not provided.
            x_component (int): Number of the component used for the x-axis.
            y_component (int): Number of the component used for the y-axis.
            show_row_points (bool): Whether to show row principal components or not.
            row_points_size (float): Row principal components point size.
            row_points_alpha (float): Alpha for the row principal component.
            show_row_labels (bool): Whether to show row labels or not.
            show_column_points (bool): Whether to show column principal components or not.
            column_points_size (float): Column principal components point size.
            show_column_labels (bool): Whether to show column labels or not.
            legend_n_cols (int): Number of columns used for the legend.
        Returns:
            matplotlib.Axis
        """

        utils.validation.check_is_fitted(self)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add style
        ax = FAMD_utility.stylize_axis(ax)

        # Plot row principal coordinates
        if show_row_points or show_row_labels:

            row_coords = self.row_coordinates(X)

            if show_row_points:
                ax.scatter(
                    row_coords.iloc[:, x_component],
                    row_coords.iloc[:, y_component],
                    s=row_points_size,
                    label=None,
                    color=self.GRAY['dark'],
                    alpha=row_points_alpha
                )

            if show_row_labels:
                for _, row in row_coords.iterrows():
                    ax.annotate(row.name, (row[x_component], row[y_component]))

        # Plot column principal coordinates
        if show_column_points or show_column_labels:

            col_coords = self.column_coordinates(X)
            x = col_coords[x_component]
            y = col_coords[y_component]

            prefixes = col_coords.index.str.split('_').map(lambda x: x[0])

            for prefix in prefixes.unique():
                mask = prefixes == prefix

                if show_column_points:
                    ax.scatter(x[mask], y[mask], s=column_points_size, label=prefix)

                if show_column_labels:
                    for i, label in enumerate(col_coords[mask].index):
                        ax.annotate(label, (x[mask][i], y[mask][i]))

            ax.legend(ncol=legend_n_cols)

        # Text
        ax.set_title('Row and column principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

        return ax
    
    
class MFA(PCA):

    def __init__(self, groups=None, normalize=True, n_components=2, n_iter=10,
                 copy=True, check_input=True, random_state=None, engine='auto'):
        super().__init__(
            rescale_with_mean=False,
            rescale_with_std=False,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine
        )
        self.groups = groups
        self.normalize = normalize

    def fit(self, X, y=None):

        # Checks groups are provided
        if self.groups is None:
            raise ValueError('Groups have to be specified')

        # Check input
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])

        # Prepare input
        X = self._prepare_input(X)

        # Check group types are consistent
        self.all_nums_ = {}
        for name, cols in sorted(self.groups.items()):
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError('Not all columns in "{}" group are of the same type'.format(name))
            self.all_nums_[name] = all_num

        # Run a factor analysis in each group
        self.partial_factor_analysis_ = {}
        for name, cols in sorted(self.groups.items()):
            if self.all_nums_[name]:
                fa = PCA(
                    rescale_with_mean=False,
                    rescale_with_std=False,
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=True,
                    random_state=self.random_state,
                    engine=self.engine
                )
            else:
                fa = MCA(
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=self.copy,
                    random_state=self.random_state,
                    engine=self.engine
                )
            self.partial_factor_analysis_[name] = fa.fit(X.loc[:, cols])

        # Fit the global PCA
        super().fit(self._build_X_global(X))

        return self

    def _prepare_input(self, X):

        # Make sure X is a DataFrame for convenience
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Copy data
        if self.copy:
            X = X.copy()

        if self.normalize:
            # Scale continuous variables to unit variance
            num = X.select_dtypes(np.number).columns
            # If a column's cardinality is 1 then it's variance is 0 which can
            # can cause a division by 0
            normalize = lambda x: x / (np.sqrt((x ** 2).sum()) or 1)
            X.loc[:, num] = (X.loc[:, num] - X.loc[:, num].mean()).apply(normalize, axis='rows')

        return X

    def _build_X_global(self, X):
        X_partials = []

        for name, cols in sorted(self.groups.items()):
            X_partial = X.loc[:, cols]

            # Dummify if there are categorical variable
            if not self.all_nums_[name]:
                X_partial = pd.get_dummies(X_partial)

            X_partials.append(X_partial / self.partial_factor_analysis_[name].s_[0])

        X_global = pd.concat(X_partials, axis='columns')
        X_global.index = X.index
        return X_global

    def transform(self, X):
        """Returns the row principal coordinates of a dataset."""
        return self.row_coordinates(X)

    def _row_coordinates_from_global(self, X_global):
        """Returns the row principal coordinates."""
        return len(X_global) ** 0.5 * super().row_coordinates(X_global)

    def row_coordinates(self, X):
        """Returns the row principal coordinates."""
        utils.validation.check_is_fitted(self)

        # Check input
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])

        # Prepare input
        X = self._prepare_input(X)

        return self._row_coordinates_from_global(self._build_X_global(X))

    def row_contributions(self, X):
        """Returns the row contributions towards each principal component."""
        utils.validation.check_is_fitted(self)

        # Check input
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])

        # Prepare input
        X = self._prepare_input(X)

        return super().row_contributions(self._build_X_global(X))

    def partial_row_coordinates(self, X):
        """Returns the row coordinates for each group."""
        utils.validation.check_is_fitted(self)

        # Check input
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])

        # Prepare input
        X = self._prepare_input(X)

        # Define the projection matrix P
        P = len(X) ** 0.5 * self.U_ / self.s_

        # Get the projections for each group
        coords = {}
        for name, cols in sorted(self.groups.items()):
            X_partial = X.loc[:, cols]

            if not self.all_nums_[name]:
                X_partial = pd.get_dummies(X_partial)

            Z_partial = X_partial / self.partial_factor_analysis_[name].s_[0]
            coords[name] = len(self.groups) * (Z_partial @ Z_partial.T) @ P

        # Convert coords to a MultiIndex DataFrame
        coords = pd.DataFrame({
            (name, i): group_coords.loc[:, i]
            for name, group_coords in coords.items()
            for i in range(group_coords.shape[1])
        })

        return coords

    def column_correlations(self, X):
        """Returns the column correlations."""
        utils.validation.check_is_fitted(self)

        X_global = self._build_X_global(X)
        row_pc = self._row_coordinates_from_global(X_global)

        return pd.DataFrame({
            component: {
                feature: row_pc[component].corr(X_global[feature])
                for feature in X_global.columns
            }
            for component in row_pc.columns
        }).sort_index()

    def plot_partial_row_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                                     color_labels=None, **kwargs):
        """Plot the row principal coordinates."""
        utils.validation.check_is_fitted(self)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add plotting style
        ax = FAMD_utility.stylize_axis(ax)

        # Check input
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])

        # Prepare input
        X = self._prepare_input(X)

        # Retrieve partial coordinates
        coords = self.partial_row_coordinates(X)

        # Determine the color of each group if there are group labels
        if color_labels is not None:
            colors = {g: ax._get_lines.get_next_color() for g in sorted(list(set(color_labels)))}

        # Get the list of all possible markers
        marks = itertools.cycle(list(markers.MarkerStyle.markers.keys()))
        next(marks)  # The first marker looks pretty shit so we skip it

        # Plot points
        for name in self.groups:

            mark = next(marks)

            x = coords[name][x_component]
            y = coords[name][y_component]

            if color_labels is None:
                ax.scatter(x, y, marker=mark, label=name, **kwargs)
                continue

            for color_label, color in sorted(colors.items()):
                mask = np.array(color_labels) == color_label
                label = '{} - {}'.format(name, color_label)
                ax.scatter(x[mask], y[mask], marker=mark, color=color, label=label, **kwargs)

        # Legend
        ax.legend()

        # Text
        ax.set_title('Partial row principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

        return ax
    
    
class FAMD(MFA):
    def __init__(self, n_components=2, n_iter=3, copy=True, check_input=True, random_state=None,
                 engine='auto'):
        super().__init__(
            groups=None,
            normalize=True,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine
        )

    def fit(self, X, y=None):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Separate numerical columns from categorical columns
        num_cols = X.select_dtypes(np.number).columns.tolist()
        cat_cols = list(set(X.columns) - set(num_cols))

        # Make one group per variable type
        self.groups = {}
        if num_cols:
            self.groups['Numerical'] = num_cols
        else:
            raise ValueError('FAMD works with categorical and numerical data but ' +
                             'you only have categorical data; you should consider using MCA')
        if cat_cols:
            self.groups['Categorical'] = cat_cols
        else:
            raise ValueError('FAMD works with categorical and numerical data but ' +
                             'you only have numerical data; you should consider using PCA')

        return super().fit(X)





class MDLP(object):
    '''
    Entropy-based Minimum description length principle.
    '''
    def discretize_feature(self, x, binning):
        '''
        Discretize a feature x with respective to the given binning
        '''
        x_discrete = [1 for i in range(len(x))]
        for i in range(len(x)):
            for cut_value in binning:
                if x[i] > cut_value:
                    x_discrete[i] = x_discrete[i] + 1
        return np.array(x_discrete)

    def target_table(self, target):
        '''
        Create a numpy array that counts the occurrences
        of values of the input vector
        Example:
        target_table([1,2,2,3,4,5,5,5,5,6])
        >>> array([1,2,1,1,4,1])
        '''
        return np.unique(target, return_counts=True)[1]

    def stable_log(self, input):
        '''
        Stable version of natural logarithm, which
        replaces elements smaller than 1*e^(-10) by
        one to avoid infinite values, then applies log as usual.
        The input variable has to be a numpy array.
        Example:
        stable_log([0,1,2])
        >>> array([1,2,3,4,5,6])
        '''
        copy = input.copy()
        copy[copy <= 1e-10] = 1
        return np.log(copy)
        
    def entropy(self, variable):
        '''
        Compute the Shannon entropy of the input variable
        Example:
        stable_log(np.array([0,1,2]))
        >>> array([0., 0., 0.69314718])
        '''
        prob = self.target_table(variable) / len(variable)
        ent = -sum(prob * self.stable_log(prob))
        return ent

    def levels(self, variable):
        '''
        Create a numpy array that lists each value of the
        input vector once.
        Example:
        levels([1,2,2,3,4,5,5,5,5,6]) >>> azΩ
        '''
        return np.unique(variable)

    def stopping_criterion(self, cut_idx, target, ent):
        '''
        Stopping criterion of the MDLP algorithm. Specifying a
        cutting index cut_idx, a target vector and the current entropy,
        the function will compute the entropy of the vector split by
        the cutting point.
        If the gain in further splitting, i.e. the decrease in entropy
        is too small, the algorithm will return "None" and MDLP will
        be stopped.
        '''
        n = len(target)
        target_entropy = self.entropy(target)
        gain = target_entropy - ent
        
        k = len(np.unique(target))
        k1 = len(np.unique(target[: cut_idx]))
        k2 = len(np.unique(target[cut_idx: ]))
        
        delta = (np.log(3**k - 2) - (k * target_entropy
                 - k1 * self.entropy(target[: cut_idx])
                 - k2 * self.entropy(target[cut_idx: ])))
        cond = np.log(n - 1) / n + delta / n
        if gain >= cond:
            return gain
        else:
            return None

    def find_cut_index(self, x, y):
        '''
        Determine the optimal cutting point (in the sense
        of minimizing entropy) for a feature vector x and
        a corresponding target vector y.
        The function will return the index of this point
        and the respective entropy.
        '''
        n = len(y)
        init_entropy = 9999
        current_entropy = init_entropy
        index = None
        for i in range(n - 1):
            if (x[i] != x[i+1]):
                cut = (x[i] + x[i + 1]) / 2.
                cutx = bisect_right(x, cut)
                weight_cutx = cutx / n
                left_entropy = weight_cutx * self.entropy(y[: cutx])
                right_entropy = (1 - weight_cutx) * self.entropy(y[cutx: ])
                temp = left_entropy + right_entropy
                if temp < current_entropy:
                    current_entropy = temp
                    index = i + 1
        if index is not None:
            return [index, current_entropy]
        else:
            return None

    def cut_points(self, x, y):
        '''
        Main function for the MDLP algorithm. A feature vector x
        and a target vector y are given as input, the algorithm
        computes a list of cut-values used for binning the variable x.
        '''
        sorted_index = np.argsort(x)
        xo = x[sorted_index]
        yo = y[sorted_index]
        depth = 1

        def getIndex(low, upp, depth=depth):
            x = xo[low:upp]
            y = yo[low:upp]
            cut = self.find_cut_index(x, y)
            if cut is None:
                return None
            cut_index = int(cut[0])
            current_entropy = cut[1]
            ret = self.stopping_criterion(cut_index, np.array(y),
                                          current_entropy)
            if ret is not None:
                return [cut_index, depth + 1]
            else:
                return None

        def part(low=0, upp=len(xo)-1, cut_points=np.array([]), depth=depth):
            x = xo[low: upp]
            if len(x) < 2:
                return cut_points
            cc = getIndex(low, upp, depth=depth)
            if (cc is None):
                return cut_points
            ci = int(cc[0])
            depth = int(cc[1])
            cut_points = np.append(cut_points, low + ci)
            cut_points = cut_points.astype(int)
            cut_points.sort()
            return (list(part(low, low + ci, cut_points, depth=depth))
                    + list(part(low + ci + 1, upp, cut_points, depth=depth)))

        res = part(depth=depth)
        cut_index = None
        cut_value = []
        if res is not None:
                cut_index = res
                for indices in cut_index:
                        cut_value.append((xo[indices-1] + xo[indices])/2.0)
        result = np.unique(cut_value)
        return result
    

def random_walk_with_outliers(origin, n_steps, perc_outliers=0.0, outlier_mult=10, seed=42):
    assert (perc_outliers >= 0.0) & (perc_outliers <= 1.0)
    #set seed for reproducibility
    np.random.seed(seed)
    
    # possible steps
    steps = [-1, 1]

    # simulate steps
    steps = np.random.choice(a=steps, size=n_steps-1)
    rw = np.append(origin, steps).cumsum(0)
    
    # add outliers
    n_outliers = int(np.round(perc_outliers * n_steps, 0))
    indices = np.random.randint(0, len(rw), n_outliers)
    rw[indices] = rw[indices] + steps[indices + 1] * outlier_mult
    return rw, indices



def datawig_simple(df3,target_columns):
    import warnings
    from tqdm import tqdm
    warnings.filterwarnings("ignore")
    for target in target_columns:
        print('woring on column: '+target)
        columns = list(df3)
        columns.remove(target)
        N = len(df3)
        # batchsize is controlled to be 10000 because of the model restriction
        sequence = [i for i in range(0,N,10000)] 
        if sequence[-1]!=N:
            sequence += [N]
        for n in tqdm(range(len(sequence)-1)):
            if n != len(sequence)-2:
                min_batch = df3.iloc[sequence[n]:sequence[n+1],:]
                imputer = datawig.SimpleImputer(input_columns=columns,
                                                output_column=target)
                imputer.fit(train_df=min_batch.dropna())
                df3.iloc[sequence[n]:sequence[n+1],list(df3).index(target)] = imputer.predict(min_batch).loc[:,target+'_imputed']
            else:
                min_batch = df3.iloc[max(0,sequence[n+1]-10000):sequence[n+1],:]
                imputer = datawig.SimpleImputer(input_columns=columns,
                                                output_column=target)
                imputer.fit(train_df=min_batch.dropna())
                df3.iloc[max(0,sequence[n+1]-10000):sequence[n+1],list(df3).index(target)] = imputer.predict(min_batch).loc[:,target+'_imputed']
    warnings.filterwarnings("default")
    return df3


def entropy_numpy(data_classes, base=2):
    from math import log
    classes = np.unique(data_classes)
    N = len(data_classes)
    ent = 0  # initialize entropy

    # iterate over classes
    for c in classes:
        partition = data_classes[data_classes == c]  # data with class = c
        proportion = len(partition) / N
        #update entropy
        ent -= proportion * log(proportion, base)

    return ent

def cut_point_information_gain_numpy(X, y, cut_point):
    entropy_full = entropy_numpy(y)  # compute entropy of full dataset (w/o split)

    #split data at cut_point
    data_left_mask = X <= cut_point #dataset[dataset[feature_label] <= cut_point]
    data_right_mask = X > cut_point #dataset[dataset[feature_label] > cut_point]
    (N, N_left, N_right) = (len(X), data_left_mask.sum(), data_right_mask.sum())

    gain = entropy_full - (N_left / N) * entropy_numpy(y[data_left_mask]) - \
        (N_right / N) * entropy_numpy(y[data_right_mask])

    return gain