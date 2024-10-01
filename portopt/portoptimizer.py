import logging

import numpy as np
import cvxopt
from typing import List, Optional, Tuple
class PortOptimizer:
    ''' Portfolio Optimization '''
    def __init__(self ,
                    VCV: np.ndarray,
                    initial_weights: np.ndarray,
                    alpha: np.ndarray,
                    txn_costs: np.ndarray,
                    shorting_costs: np.ndarray,
                    risk_multiplier: float,
                    alpha_multiplier: float,
                    txn_cost_multiplier: float,
                    shorting_cost_multiplier: float,
                    max_concentration: float = 0.2,
                    weight_bounds: Optional[List[Optional[Tuple[float,float]]]] = None,
                    max_iterations: int = 100
                    ):
        ''' Initialize the Portfolio Optimization problem
            @param VCV: Variance-Covariance Matrix of the assets
            @param initial_weights: Initial weights of the assets
            @param alpha: Expected returns of the assets
            @param txn_costs: Linear transaction costs ( per unit traded ) of the assets
            @param shorting_costs: Linear shorting costs (per unit held short ) of the assets
            @param risk_multiplier: Risk aversion parameter in the objective function
            @param alpha_multiplier: Alpha multiplier parameter in the objective function
            @param txn_cost_multiplier: Transaction cost multiplier parameter in the objective function
            @param shorting_cost_multiplier: Shorting cost multiplier parameter in the objective function
            @param max_concentration: Maximum concentration of any asset in the portfolio on long or short side
            @param weight_bounds: List of tuples (lower,upper) for each asset weight. None means no bound
        '''
        assert VCV is not None and VCV.shape[0] == VCV.shape[1]
        self.num_assets = VCV.shape[0]
        self.VCV = VCV
        self.initial_weights = initial_weights if initial_weights is not None else np.zeros(self.num_assets)
        self.alpha = alpha if alpha is not None else np.zeros(self.num_assets)
        self.txn_costs = txn_costs if txn_costs is not None else np.zeros(self.num_assets)
        self.shorting_costs = shorting_costs if shorting_costs is not None else np.zeros(self.num_assets)
        self.risk_multiplier = risk_multiplier if risk_multiplier is not None else 1.0
        self.alpha_multiplier = alpha_multiplier if alpha_multiplier is not None else 1.0
        self.txn_cost_multiplier = txn_cost_multiplier if txn_cost_multiplier is not None else 1.0
        self.shorting_cost_multiplier = shorting_cost_multiplier if shorting_cost_multiplier is not None else 1.0
        self.max_concentration = max_concentration if max_concentration is not None else 0.2
        self.weight_bounds = weight_bounds if weight_bounds is not None else [None]*self.num_assets
        self.max_iterations = max_iterations

        if np.max(np.abs(self.initial_weights)) > self.max_concentration:
            #cvxopt cannot handle infeasible problems, so we adjust max_concentration to be at least as large as the maximum initial weight
            logging.warning('Initial weights exceed maximum concentration - adjusting max_concentration from %.2f to %.2f', \
                            self.max_concentration, float(np.max(np.abs(self.initial_weights))))
            self.max_concentration = float(np.max(np.abs(self.initial_weights)))
        assert self.initial_weights.shape == (self.num_assets,)
        assert self.alpha.shape == (self.num_assets,)
        assert self.txn_costs.shape == (self.num_assets,)
        assert self.shorting_costs.shape == (self.num_assets,)

    def solve(self, show_progress = False,maxiters = None, feastol=1e-9, abstol=1e-9 ) -> dict:
        ''' Solve the portfolio optimization problem
            @return: Optimal portfolio positions, optimal portfolio trades
        '''
        maxiters = maxiters if maxiters is not None else self.max_iterations
        cvx_mat = self._setup_matrices()
        result = cvx_mat.copy()
        P,q,G,h,A,b = cvx_mat['P'], cvx_mat['q'], cvx_mat['G'], cvx_mat['h'], cvx_mat['A'], cvx_mat['b']
        cvxopt.solvers.options['show_progress'] = show_progress
        cvxopt.solvers.options['maxiters'] = maxiters
        cvxopt.solvers.options['feastol'] = feastol
        cvxopt.solvers.options['abstol'] = abstol

        sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))
        if sol['status'] != 'optimal':
            raise Exception("Optimization did not converge - status: %s" % sol['status'])
        else:
            solx = np.array(sol['x']).flatten() # Optimal solution X=( Lb, Ls, Sb, Ss )
            DEBUG=False
            if DEBUG:
                solRisk = solx@cvx_mat['riskQuadratic']@solx + cvx_mat['riskLinear']@solx + cvx_mat['riskConstant']
                solAlpha = solx@cvx_mat['alpha'] + cvx_mat['alpha_const']
                solTxnCost = solx@cvx_mat['txn_cost']
                solShortingCost = solx@cvx_mat['shorting_cost'] + cvx_mat['shorting_cost_const']
                objective = self.risk_multiplier* solRisk - self.alpha_multiplier * solAlpha + self.txn_cost_multiplier * solTxnCost + self.shorting_cost_multiplier * solShortingCost
                logging.warning('SOLUTION Objective: %.4f Risk: %.4f, Alpha: %.4f (%.4f+%.4f), TxnCost: %.4f, ShortingCost: %.4f',
                                objective, solRisk, solAlpha, cvx_mat['alpha_const'],solx@cvx_mat['shorting_cost'],solTxnCost, solShortingCost)
                solz = np.zeros_like(solx)
                solZRisk = solz @ cvx_mat['riskQuadratic'] @ solz + cvx_mat['riskLinear'] @ solz + cvx_mat[
                    'riskConstant']
                solZAlpha = solz @ cvx_mat['alpha'] + cvx_mat['alpha_const']
                solZTxnCost = solz @ cvx_mat['txn_cost']
                solZShortingCost = solz @ cvx_mat['shorting_cost'] + cvx_mat['shorting_cost_const']
                zobjective = self.risk_multiplier * solZRisk - self.alpha_multiplier * solZAlpha + self.txn_cost_multiplier * solZTxnCost + self.shorting_cost_multiplier * solZShortingCost
                logging.warning('ZERO SOLUTION Objective: %.4f Risk: %.4f, Alpha: %.4f (%.4f+%.4f), TxnCost: %.4f, ShortingCost: %.4f',
                                zobjective, solZRisk, solZAlpha, solZTxnCost,cvx_mat['alpha_const'],solz @ cvx_mat['alpha'], solZShortingCost)

            result['sol'] = solx
            _ext = lambda i: solx[self.num_assets*i:self.num_assets*(i+1)]
            ( Lb, Ls, Sb, Ss) = (_ext(0), _ext(1), _ext(2), _ext(3))
            optPositions = self.initial_weights + Lb - Ls + Sb - Ss
            if np.max(np.abs(optPositions)) > self.max_concentration:
                logging.warning('Optimization broke the max_conentration constraint !!!' )
                raise Exception('FUBAR')
            optTrades = Lb + Ls - Sb - Ss
            result['optPositions'] = optPositions; result['optTrades'] = optTrades
            return result

    def _decoder_matrix(self, N, a, b, c, d):
        result = np.zeros((N, 4 * N))
        for i in range(N):
            result[i,i]=a
            result[i, N + i] = b
            result[i, 2 * N + i] = c
            result[i, 3 * N + i] = d

        return result

    def _setup_matrices(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        ''' Set up the CVXOPT problem matries
            returns P,q,G,h,A,b the for problem of minimize    (1/2)*x'*P*x + q'*x, subject to G*x <= h, A*x = b
        '''
        # Optimization variables will be a stacked vector   X=( BL, SL, BS, SS ), where:
        # BL = Buy while staying long ( positive )
        # SL = Sell while staying long ( positive )
        # BS = Buy while staying short ( positive )
        # SS = Sell while staying short ( positive )

        # variable bounds ( assume W0 is known initial signed position vector ):
        # if W0 >=0 then:
        #       0 <= BL <= max_concentration-W0
        #       0 <= SL <= W0
        #       0 <= BS <= 0
        #       0 <= SS <= max_concentration
        # if W0 < 0 then:
        #       0 <= BL <= max_concentration
        #       0 <= SL <= 0
        #       0 <= BS <= -1*W0
        #       0 <= SS <= max_concentration+W0

        # Putting both above cases together:
        #       0 <= BL <= max_concentration - max(W0,0)
        #       0 <= SL <= max(W0,0)
        #       0 <= BS <= -1*min(W0,0)
        #       0 <= SS <= max_concentration + min(W0,0)

        # Resulting positions is W0 + BL - SL + BS - SS.
        # if W0 is >= 0 then
        #   Long positions is W0+BL-SL, Short positions is SS (hoping that SL will be maxed to W0 first before SS)
        # if W0 is < 0 then
        #   Long positions is BL, Short positions is W0-BS+SS
        # But recall that when W0>=0, BS is 0 by constraint. And W0<0 then SL is 0 by constraint
        # short position is SS - BS - min(W0,0) (W0 is signed, SS and BS not)
        # long positions is BL - SL + max(W0,0) (W0 is signed, SL is not)

        #risk term let T = BL-SL+BS-SS
        #risk term = (W0+T)'*C*(W0+T) = W0'*C*W0 + T'*C*T + 2*W0'*C*T
        # W0'*C*W0 is constant - goes away when minimizing
        # T'*C*T = [(BL - SL)_t + (BS - SS)_t ] * C * [(BL-SL)+(BS-SS)]
        # assume BLSL is stacked ( BL,SL), BSSS is stacked (BS,SS) and M is (C -C; -C C) matrix. Then above is
        # BLSL_t * M * BLSL + BSSS_t * M * BSSS + 2 * (BL-SL)_t * C * (BS-SS) =
        # BLSL_t * M * BLSL + BSSS_t * M * BSSS + 2 * BLSL_t * M * BSSS
        # = BLSLBSSS_t * M2 * BLSLBSSS where M2 is ( M M ; M M ) [Eq 1]
        # 2*W0'*C*T  = 2*W0'*C*(BL-SL+BS-SS) = 2 * W0'*C*(1 -1 1 -1) * BLSLBSSS

        #so if X is our stacked vector (BL SL BS SS ) then our risk (variance) function is
        # X' * M2 * X + 2[W0'*C*(1 -1 1 -1)] * X + W0'*C*W0
        #
        # Resulting trades is BL + SL + BS + SS
        # Variable relationships
        # sum(BL+SL) = 1-W0
        # sum(BS+SS) = 1-W0


        # Objective function:
        # Minimize risk_multiplier * port_variance   - alpha_multiplier * port_alpha + txn_cost_multiplier * txn_costs + shorting_cost_multiplier * shorting_costs
        # where:
        # port_variance = (L,S,....)_transpose * VCV * (L,S,....
        result = {}
        NV = 4 * self.num_assets
        NA = self.num_assets
        # Risk 1/2 * X' * M2 * X + [W0'*C*(1 -1 1 -1)] * X
        # X is stacked vector (BL SL BS SS )

        M = np.r_[np.c_[self.VCV,-self.VCV],np.c_[-self.VCV,self.VCV]]
        riskQuadratic = np.r_[np.c_[M,M],np.c_[M,M]]
        riskLinear =  2*self.initial_weights.T@(self.VCV)@self._decoder_matrix(NA, 1,-1,1,-1)
        riskConstant = self.initial_weights.T@(self.VCV)@self.initial_weights

        result['riskQuadratic'] = riskQuadratic; result['riskLinear'] = riskLinear; result['riskConstant'] = riskConstant

        #X is our stacked vector (BL SL BS SS )
        alpha = self.alpha@self._decoder_matrix(NA, 1,-1,1,-1)
        txn_cost = self.txn_costs@self._decoder_matrix(NA, 1,1,1,1)
        # short position is SS - BS - min(W0,0) (W0 is signed, SS and BS not)
        # long positions is BL - SL + max(W0,0) (W0 is signed, SL is not)
        shorting_cost = self.shorting_costs@self._decoder_matrix(NA, 0,0,-1,1)
        result['alpha'] = alpha; result['txn_cost'] = txn_cost; result['shorting_cost'] = shorting_cost
        result['alpha_const'] = self.alpha.dot(self.initial_weights)
        result['shorting_cost_const'] = -1*np.minimum(self.initial_weights,0).dot(self.shorting_costs)

        P = self.risk_multiplier * riskQuadratic;
        q = self.risk_multiplier * riskLinear + -1*self.alpha_multiplier*alpha + self.txn_cost_multiplier*txn_cost + \
                                                      self.shorting_cost_multiplier * shorting_cost
        result['P'] = P; result['q'] = q
        # Inequality constraints ( Gx <= h )
        #       0 <= BL <= max_concentration - max(W0,0)
        #       0 <= SL <= max(W0,0)
        #       0 <= BS <= -1*min(W0,0)
        #       0 <= SS <= max_concentration + min(W0,0)

        upperBoundLeft = np.diag(np.ones(NV))
        upperBoundRight = np.zeros(NV)
        for i in range(NA):
            assetWeightBounds = self.weight_bounds[i]
            lowBound = assetWeightBounds[0] if assetWeightBounds is not None else -np.inf
            hiBound = assetWeightBounds[1] if assetWeightBounds is not None else np.inf
            max_long_position = min(self.max_concentration,hiBound)
            max_short_position = max(-1*self.max_concentration,lowBound)
            upperBoundRight[i]=max_long_position - max(self.initial_weights[i],0)
            upperBoundRight[NA+i] = max(self.initial_weights[i],0)
            upperBoundRight[2*NA+i] = -1*min(self.initial_weights[i],0)
            upperBoundRight[3*NA+i] = -max_short_position + min(self.initial_weights[i],0)

        lowerBoundLeft = -1*np.diag(np.ones(NV))
        lowerBoundRight = np.array([0]*NV)
        G = np.concatenate([upperBoundLeft, lowerBoundLeft]); h = np.concatenate([upperBoundRight, lowerBoundRight])
        result['G'] = G; result['h'] = h

        # Equality constraints ( Ax = b )
        # long positions is BL - SL + max(W0,0) (W0 is signed, SL is not)
        # short position is SS - BS - min(W0,0) (W0 is signed, SS and BS not)

        A = np.r_[
            np.ones(NA).reshape(1,NA)@self._decoder_matrix(NA, 1,-1,0,0),
            np.ones(NA).reshape(1,NA)@self._decoder_matrix(NA, 0,0,-1,1)
        ]
        b=np.array([float(1-sum(np.maximum(self.initial_weights,0))), float(1+sum(np.minimum(self.initial_weights,0)))])
        result['A'] = A; result['b'] = b
        return result




