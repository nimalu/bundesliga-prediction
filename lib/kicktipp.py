from collections import namedtuple

from robobrowser import RoboBrowser

Bet = namedtuple("Bet", "match_day match_id home_team away_team home_bet away_bet")


class KicktippApi:
    def __init__(self):
        self.browser = RoboBrowser(parser="html5lib")

    def login(self, username: str, password: str) -> None:
        self.browser.open("https://www.kicktipp.de/info/profil/login")
        form = self.browser.get_form()
        form["kennung"] = username
        form["passwort"] = password
        self.browser.submit_form(form)

    def _build_bet_url(self, community, matchday):
        return (
            f"https://www.kicktipp.de/{community}/tippabgabe?&spieltagIndex={matchday}"
        )

    def get_open_bets(self, community, matchday) -> list[Bet]:
        url = self._build_bet_url(community, matchday)
        self.browser.open(url)

        home_teams = [td.text for td in self.browser.select("tr > td:nth-child(2)")]
        away_teams = [td.text for td in self.browser.select("tr > td:nth-child(3)")]

        def value_default_0(inp):
            return int(inp.attrs["value"]) if "value" in inp.attrs else 0

        home_bets = [
            value_default_0(inp)
            for inp in self.browser.select('input[id$="_heimTipp"]')
        ]
        away_bets = [
            value_default_0(inp)
            for inp in self.browser.select('input[id$="_gastTipp"]')
        ]
        matches = []
        for i in range(len(home_teams)):
            try:
                matches.append(
                    Bet(
                        matchday,
                        i,
                        home_teams[i],
                        away_teams[i],
                        home_bets[i],
                        away_bets[i],
                    )
                )
            except Exception:
                continue
        return matches

    def submit_bets(self, community: str, bets: list[Bet]) -> None:
        bets_by_matchday: dict[str, list[Bet]] = {}
        for t in bets:
            match_day = str(t.match_day)
            if match_day not in bets_by_matchday:
                bets_by_matchday[match_day] = []
            bets_by_matchday[match_day].append(t)

        for match_day, matchday_bets in bets_by_matchday.items():
            self.browser.open(self._build_bet_url(community, match_day))
            form = self.browser.get_form()
            field_home_tips = self.browser.select('input[id$="_heimTipp"]')
            field_away_tips = self.browser.select('input[id$="_gastTipp"]')
            for match in matchday_bets:
                home_field = field_home_tips[match.match_id]
                away_field = field_away_tips[match.match_id]
                form[home_field.attrs["name"]] = str(match.home_bet)
                form[away_field.attrs["name"]] = str(match.away_bet)

            self.browser.submit_form(form, submit="submitbutton")
