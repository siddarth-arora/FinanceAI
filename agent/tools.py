"""Read-only access to a detached, validated analysis snapshot."""

from copy import deepcopy
from typing import Any

from agent.profiles import select_profiles
from src.portfolio import calculate_portfolio_volatility, calculate_sharpe_ratio


class AnalysisTools:
    def __init__(self, payload: dict[str, Any]):
        self._payload = deepcopy(payload)
        self._profiles = select_profiles(self._payload)

    def get_metadata(self) -> dict[str, Any]:
        return deepcopy(self._payload['metadata'])

    def get_profile(self, name: str) -> dict[str, Any]:
        if name not in ('low', 'medium', 'high'):
            raise ValueError('Unknown profile.')
        return self._profiles[name].to_dict()

    def get_asset_stats(self, ticker: str) -> dict[str, Any]:
        assets = self._payload['metadata']['assets']
        if ticker not in assets:
            raise ValueError('Unknown asset.')
        weights = [float(asset == ticker) for asset in assets]
        covariance = self._covariance()
        return {
            'ticker': ticker,
            'annual_expected_return': self._payload['annual_expected_returns'][ticker],
            'annual_volatility': calculate_portfolio_volatility(weights, covariance),
            'metadata': self.get_metadata(),
        }

    def get_frontier_point(self, index: int) -> dict[str, Any]:
        points = self._payload['efficient_frontier']
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(points):
            raise ValueError('Invalid frontier index.')
        point = deepcopy(points[index])
        assets = self._payload['metadata']['assets']
        point['sharpe_ratio'] = calculate_sharpe_ratio(
            [point['weights'][ticker] for ticker in assets],
            [self._payload['annual_expected_returns'][ticker] for ticker in assets],
            self._covariance(), self._payload['metadata']['risk_free_rate'],
        )
        point['metadata'] = self.get_metadata()
        return point

    def _covariance(self) -> list[list[float]]:
        assets = self._payload['metadata']['assets']
        return [[self._payload['annual_covariance'][a][b] for b in assets] for a in assets]

    @property
    def frontier_count(self) -> int:
        return len(self._payload['efficient_frontier'])

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Reject extra arguments and unknown operations; never use dynamic eval."""
        functions = {
            'get_metadata': (self.get_metadata, set()),
            'get_profile': (self.get_profile, {'name'}),
            'get_asset_stats': (self.get_asset_stats, {'ticker'}),
            'get_frontier_point': (self.get_frontier_point, {'index'}),
        }
        if name not in functions or not isinstance(arguments, dict):
            raise ValueError('Invalid read-only tool call.')
        function, required = functions[name]
        if set(arguments) != required:
            raise ValueError('Invalid read-only tool arguments.')
        if any(not isinstance(arguments[key], str) for key in required - {'index'}):
            raise ValueError('Tool names and tickers must be strings.')
        return function(**arguments)

    def definitions(self) -> list[dict[str, Any]]:
        metadata = self.get_metadata()
        fields = {
            'get_metadata': ({}, 'Read dataset provenance and constraints.'),
            'get_profile': ({'name': {'type': 'string', 'enum': ['low', 'medium', 'high']}},
                            'Read an existing sampled profile, including warnings.'),
            'get_asset_stats': ({'ticker': {'type': 'string', 'enum': metadata['assets']}},
                                'Read historical return and volatility of one asset.'),
            'get_frontier_point': ({'index': {'type': 'integer', 'minimum': 0,
                                             'maximum': self.frontier_count - 1}},
                                   'Read a sampled frontier point; index is zero-based.'),
        }
        return [
            {'type': 'function', 'name': name, 'description': description,
             'parameters': {'type': 'object', 'properties': properties,
                            'required': list(properties), 'additionalProperties': False}}
            for name, (properties, description) in fields.items()
        ]
