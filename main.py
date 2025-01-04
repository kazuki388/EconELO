from __future__ import annotations

import asyncio
import functools
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union

import aiofiles
import aioshutil
import interactions
import orjson
from interactions.client.errors import NotFound
from interactions.ext.paginators import Paginator

BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
LOG_FILE: str = os.path.join(BASE_DIR, "econelo.log")
ELO_FILE: str = os.path.join(BASE_DIR, "econelo.json")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s | %(process)d:%(thread)d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    "%Y-%m-%d %H:%M:%S.%f %z",
)
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=1024 * 1024, backupCount=1, encoding="utf-8"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Model


def format_discord_timestamp(dt: datetime) -> str:
    unix_ts = int(dt.timestamp())
    return f"<t:{unix_ts}:F> (<t:{unix_ts}:R>)"


class EmbedColor(Enum):
    OFF = 0x5D5A58
    FATAL = 0xFF4343
    ERROR = 0xE81123
    WARN = 0xFFB900
    INFO = 0x0078D7
    DEBUG = 0x00B7C3
    TRACE = 0x8E8CD8
    ALL = 0x0063B1


@dataclass
class Config:
    admin_role_id: int = 1200100104682614884
    reward_roles: tuple[int, ...] = field(
        default_factory=lambda: (
            1241802041576390687,
            1243261836187664545,
            1275980805273026571,
            1292065942544711781,
            1200052609487208488,
            1206939229418946560,
        )
    )

    log_channel_id: int = 1166627731916734504
    log_forum_id: int = 1159097493875871784
    log_post_id: int = 1325002509713936434
    status_roles: tuple[int, ...] = field(
        default_factory=lambda: (
            1200043628899356702,
            1282944839679344721,
            1164761892015833129,
        )
    )
    status_amounts: dict[int, dict[str, float]] = field(
        default_factory=lambda: {
            1200043628899356702: {
                "daily": 15.0,
            },
            1282944839679344721: {
                "daily": 10.0,
            },
            1164761892015833129: {
                "daily": 5.0,
            },
        }
    )
    guild_id: int = 1150630510696075404

    reward_amounts: dict[int, dict[str, float]] = field(
        default_factory=lambda: {
            1241802041576390687: {
                "daily": 50.0,
                "weekly": 500.0,
                "monthly": 2500.0,
                "seasonal": 8000.0,
                "yearly": 35000.0,
            },
            1243261836187664545: {
                "daily": 40.0,
                "weekly": 400.0,
                "monthly": 2000.0,
                "seasonal": 6000.0,
                "yearly": 28000.0,
            },
            1275980805273026571: {
                "daily": 35.0,
                "weekly": 350.0,
                "monthly": 1500.0,
                "seasonal": 5000.0,
                "yearly": 22000.0,
            },
            1292065942544711781: {
                "daily": 30.0,
                "weekly": 300.0,
                "monthly": 1200.0,
                "seasonal": 4000.0,
                "yearly": 18000.0,
            },
            1200052609487208488: {
                "daily": 25.0,
                "weekly": 250.0,
                "monthly": 1000.0,
                "seasonal": 3000.0,
                "yearly": 15000.0,
            },
            1206939229418946560: {
                "daily": 20.0,
                "weekly": 200.0,
                "monthly": 800.0,
                "seasonal": 2500.0,
                "yearly": 12000.0,
            },
        }
    )

    welcome_base_points: int = 5
    newbie_tasks: dict[str, dict[str, int | str]] = field(
        default_factory=lambda: {
            "read_guide": {"points": 3, "description": "Read server guide"},
            "introduce": {"points": 4, "description": "Make self-introduction"},
            "first_message": {"points": 2, "description": "Send first message"},
            "add_reaction": {"points": 1, "description": "Add reaction to a message"},
            "join_voice": {"points": 3, "description": "Join voice channel"},
            "use_bot": {"points": 2, "description": "Use bot features"},
            "join_event": {"points": 5, "description": "Participate in server event"},
            "invite_friend": {"points": 8, "description": "Invite a new member"},
        }
    )

    daily_points: int = field(default_factory=lambda: random.randint(8, 12))

    message_reward: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "base": {"min": 1, "max": 5},
            "daily_limit": {"base": 100, "variance": 50, "min": 50},
            "bonuses": {
                "length": {"threshold": 100, "points": 2},
                "image": {"points": 3},
                "link": {"points": 2},
                "entropy": {"threshold": 0.7, "points": 3},
                "engagement": {"replies": 2, "reactions": 1},
                "quality": {"multiplier": 1.5},
            },
            "penalties": {
                "spam": {"threshold": 0.9, "multiplier": 0.5},
                "repetition": {"threshold": 0.8, "multiplier": 0.7},
            },
        }
    )

    reaction_reward: dict[str, Any] = field(
        default_factory=lambda: {
            "points": 5,
            "daily_limit": 50,
            "emoji": "⭐",
            "special_emojis": {"👍": 10, "💎": 8, "🌟": 7},
        }
    )

    invite_reward: dict[str, Any] = field(
        default_factory=lambda: {
            "points": 10,
            "stay_days": 7,
            "daily_limit": 5,
            "pending_rewards": {},
            "daily_invites": {},
            "milestones": {"5": 100, "10": 250, "25": 1000},
        }
    )

    economy_control: dict[str, float] = field(
        default_factory=lambda: {
            "inflation_rate": 0.03,
            "points_decay": 0.01,
            "market_volatility": 0.05,
            "bonus_multiplier": 1.2,
            "penalty_multiplier": 0.8,
            "seasonal_boost": 1.5,
        }
    )

    community_rewards: dict[str, int] = field(
        default_factory=lambda: {
            "help_others": 5,
            "create_content": 10,
            "organize_event": 20,
            "bug_report": 3,
            "suggestion_adopted": 15,
            "mentor_newbie": 25,
            "contribute_resources": 15,
            "moderate_discussion": 10,
            "translate_content": 12,
        }
    )

    levels: dict[str, dict[str, int | str]] = field(
        default_factory=lambda: {
            "1": {
                "points": 666666,
                "title": "初出茅庐",
                "role_id": 1292379114476671048,
            },
            "2": {"points": 888888, "title": "赌神", "role_id": 1292379114476671048},
            "3": {
                "points": 999999,
                "title": "大舞台传奇",
                "role_id": 1312024848385703937,
            },
        }
    )

    activity_metrics: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "message_quality": {
                "length_weight": 0.3,
                "engagement_weight": 0.4,
                "frequency_weight": 0.3,
                "uniqueness_weight": 0.2,
                "helpfulness_weight": 0.3,
            },
            "participation": {
                "voice_time": 0.25,
                "reactions": 0.25,
                "messages": 0.25,
                "events": 0.25,
                "contributions": 0.2,
                "community_building": 0.3,
            },
            "reputation": {
                "helpful_votes": 0.4,
                "report_accuracy": 0.3,
                "moderation_actions": 0.3,
            },
        }
    )

    user_default: dict[str, Any] = field(
        default_factory=lambda: {
            "points": 0,
            "total_points": 0,
            "level": 1,
            "weekly_activity": 0,
            "last_daily": None,
            "role_status": {
                "daily": None,
                "weekly": None,
                "monthly": None,
                "seasonal": None,
            },
            "contribution_metrics": {
                "messages": 0,
                "reactions": 0,
                "voice_time": 0,
                "helps": 0,
                "events_organized": 0,
                "resources_shared": 0,
            },
            "market_participation": {
                "trades": 0,
                "items_owned": [],
                "reputation": 100,
                "trade_history": [],
                "favorite_items": [],
            },
            "achievements": [],
            "skills": {
                "leadership": 0,
                "helpfulness": 0,
                "creativity": 0,
                "engagement": 0,
            },
            "statistics": {
                "messages_sent": 0,
                "reactions_added": 0,
                "invites": 0,
                "voice_minutes": 0,
                "events_attended": 0,
                "quality_contributions": 0,
                "helpful_flags": 0,
            },
            "badges": [],
            "titles": [],
            "inventory": [],
            "streaks": {
                "daily_login": 0,
                "weekly_participation": 0,
                "helping_others": 0,
            },
        }
    )

    fed: dict[str, Any] = field(
        default_factory=lambda: {
            "initial_reserve": 99_999_999,
            "min_reserve": 5_000_000,
            "interest_rate": 0.05,
            "inflation_target": 0.02,
            "max_bet_ratio": 0.01,
            "house_edge": 0.05,
            "max_debt": -100000,
            "debt_interest_rate": 0.05,
            "debt_interest_interval": 86400,
            "debt_collection_fee": 0.10,
            "dynamic_odds": {
                "enabled": True,
                "min_multiplier": 0.5,
                "max_multiplier": 2.0,
                "adjustment_rate": 0.01,
            },
            "reward_allocation": {
                "role_rewards": 0.3,
                "daily_rewards": 0.2,
                "activity_rewards": 0.2,
                "casino_operation": 0.3,
            },
            "reward_limits": {
                "role": {
                    "daily": 1000,
                    "weekly": 5000,
                    "monthly": 25000,
                    "seasonal": 100000,
                },
                "daily_claim": 500,
                "message": 200,
                "reaction": 100,
            },
            "market_intervention": {
                "buy_threshold": 0.8,
                "sell_threshold": 1.2,
                "intervention_rate": 0.1,
            },
        }
    )

    tax_rates: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "claim": {
                "0": 0.00,
                "1000": 0.05,
                "5000": 0.10,
                "10000": 0.15,
            },
            "casino": {
                "0": 0.05,
                "1000": 0.10,
                "5000": 0.15,
                "10000": 0.20,
            },
        }
    )

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Config has no attribute `{key}`")


class Model:
    def __init__(self) -> None:
        self.elo: dict[str, Any] = {}
        self.cfg = Config()
        self.fed_state: dict[str, Any] = {
            "reserve": self.cfg.fed["initial_reserve"],
            "total_supply": 0,
            "interest_rate": self.cfg.fed["interest_rate"],
            "total_bets": 0,
            "total_payouts": 0,
            "current_odds": 1.0,
            "daily_emissions": {
                "role_rewards": 0,
                "daily_rewards": 0,
                "activity_rewards": 0,
                "casino_payouts": 0,
            },
            "last_reset": datetime.now(timezone.utc).date().isoformat(),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

    async def load_elo(self) -> None:
        try:
            if not os.path.exists(ELO_FILE):
                self.elo = {}
                await self.save_elo()
                return

            async with aiofiles.open(ELO_FILE, mode="rb") as f:
                self.elo = orjson.loads(await f.read())

        except (IOError, orjson.JSONDecodeError) as e:
            logger.error("Failed to load ELO data: %r", e)
            raise
        except Exception as e:
            logger.critical("Unexpected error loading ELO data: %r", e, exc_info=True)
            raise e

    async def save_elo(self) -> None:
        try:
            opts = (
                orjson.OPT_SERIALIZE_NUMPY
                | orjson.OPT_SERIALIZE_DATACLASS
                | orjson.OPT_INDENT_2
            )

            serialized = orjson.dumps(
                self.elo, option=opts, default=lambda x: x.__dict__
            )

            dirname = os.path.dirname(ELO_FILE)
            os.makedirs(dirname, exist_ok=True)

            temp_file = f"{ELO_FILE}.tmp"
            async with aiofiles.open(temp_file, mode="wb") as f:
                await f.write(serialized)
            os.replace(temp_file, ELO_FILE)

        except (IOError, orjson.JSONEncodeError) as e:
            logger.error("Failed to save ELO data: %r", e)
            raise

    async def get_user_elo(self, user_id: str) -> dict[str, Any]:
        return self.elo.setdefault("users", {}).get(str(user_id), self.cfg.user_default)

    async def get_user_daily_reaction_points(self, user_id: str) -> int:
        user_data = await self.get_user_elo(user_id)
        today = datetime.now(timezone.utc).date().isoformat()
        return (
            lambda r: (
                r["points"]
                if r["date"] == today
                else (r.update({"date": today, "points": 0}) or 0)
            )
        )(user_data.setdefault("daily_reactions", {"date": today, "points": 0}))

    async def update_user_elo(self, user_id: str, data: dict) -> None:
        try:

            def update_nested(target: dict, source: dict) -> None:
                for key, value in source.items():
                    if (
                        isinstance(value, dict)
                        and key in target
                        and isinstance(target[key], dict)
                    ):
                        update_nested(target[key], value)
                    else:
                        target[key] = value

            users = self.elo.setdefault("users", {})
            user_data = users.setdefault(str(user_id), {})
            update_nested(user_data, data)
            await self.save_elo()

        except Exception as e:
            logger.error(f"Error updating user ELO: {e}", exc_info=True)
            raise

    async def can_emit_points(self, reward_type: str, amount: int) -> bool:
        try:
            fed_state = self.fed_state
            federal_reserve = self.cfg.fed
            reward_limits = federal_reserve["reward_limits"]
            daily_emissions = fed_state["daily_emissions"]
            today = datetime.now(timezone.utc).date().isoformat()
            if fed_state["last_reset"] != today:
                fed_state["daily_emissions"] = {k: 0 for k in daily_emissions}
                fed_state["last_reset"] = today

            if fed_state["reserve"] - amount < federal_reserve["min_reserve"]:
                return False

            if reward_type in reward_limits:
                current_emissions = daily_emissions.get(f"{reward_type}_rewards", 0)
                limit = reward_limits.get(reward_type, 0)
                if (
                    isinstance(limit, (int, float))
                    and current_emissions + amount > limit
                ):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking point emission: {e}", exc_info=True)
            return False

    async def update_market_state(self) -> None:
        try:
            now = datetime.now(timezone.utc)
            fed_state = self.fed_state
            federal_reserve = self.cfg.fed

            reserve_ratio = fed_state["reserve"] / federal_reserve["initial_reserve"]
            market_intervention = federal_reserve["market_intervention"]
            intervention_rate = market_intervention["intervention_rate"]

            fed_state["interest_rate"] *= (
                1 + intervention_rate
                if reserve_ratio < market_intervention["buy_threshold"]
                else (
                    1 - intervention_rate
                    if reserve_ratio > market_intervention["sell_threshold"]
                    else 1
                )
            )

            if federal_reserve["dynamic_odds"]["enabled"]:
                total_bets = fed_state["total_bets"] or 1
                profit_ratio = (
                    fed_state["total_bets"] - fed_state["total_payouts"]
                ) / total_bets
                dynamic_odds = federal_reserve["dynamic_odds"]
                adjustment = dynamic_odds["adjustment_rate"] * (
                    -1 if profit_ratio < federal_reserve["house_edge"] else 1
                )

                fed_state["current_odds"] = min(
                    dynamic_odds["max_multiplier"],
                    max(
                        dynamic_odds["min_multiplier"],
                        fed_state["current_odds"] * (1 + adjustment),
                    ),
                )

            fed_state["last_update"] = now.isoformat()
            await self.save_elo()

        except Exception as e:
            logger.error(f"Failed to update market state: {e}", exc_info=True)
            raise

    async def emit_points(
        self,
        user_id: str,
        amount: int,
        reward_type: str,
        reason: str,
        apply_tax: bool = True,
    ) -> bool:
        try:
            if not await self.can_emit_points(reward_type, amount):
                return False

            final_amount = amount
            tax_amount = 0

            if apply_tax:
                final_amount, tax_amount = await self.calculate_tax(amount, "claim")

            user_data = await self.get_user_elo(user_id)
            points, total_points = (
                user_data.get(k, 0) for k in ("points", "total_points")
            )

            user_data |= dict(
                points=points + final_amount, total_points=total_points + final_amount
            )

            fed_state = self.fed_state
            fed_state.update(
                {
                    "reserve": fed_state["reserve"] - amount + tax_amount,
                    "total_supply": fed_state["total_supply"] + amount,
                    "tax_collected": fed_state.get("tax_collected", 0) + tax_amount,
                    "daily_emissions": {
                        **fed_state["daily_emissions"],
                        f"{reward_type}_rewards": fed_state["daily_emissions"].get(
                            f"{reward_type}_rewards", 0
                        )
                        + amount,
                    },
                }
            )

            await self.update_user_elo(user_id, user_data)
            await self.save_elo()

            await self.log_points_transaction(
                user_id,
                final_amount,
                f"{reason}{' (Tax: ' + f'{tax_amount:,}' + ')' if tax_amount else ''}",
                reward_type,
            )

            return True

        except Exception as e:
            logger.error(f"Error emitting points: {e}", exc_info=True)
            return False

    async def calculate_tax(self, amount: int, tax_type: str) -> tuple[int, int]:
        try:
            if not (tax_brackets := self.cfg.tax_rates.get(tax_type)):
                return amount, 0

            sorted_brackets = tuple(
                sorted(
                    map(lambda x: (int(x[0]), x[1]), tax_brackets.items()),
                    key=lambda x: x[0],
                )
            )

            thresholds = tuple(t[0] for t in sorted_brackets)
            rates = tuple(t[1] for t in sorted_brackets)
            next_thresholds = thresholds[1:] + (float("inf"),)

            taxable_amounts = tuple(
                map(
                    lambda x: min(amount - x[0], x[1] - x[0]),
                    zip(thresholds, next_thresholds),
                )
            )

            taxes = tuple(
                map(
                    lambda x: int(x[0] * x[1]) if x[0] > 0 else 0,
                    zip(taxable_amounts, rates),
                )
            )

            total_tax = sum(taxes)
            return amount - total_tax, total_tax

        except Exception as e:
            logger.error(f"Error calculating tax: {e}", exc_info=True)
            return amount, 0

    async def apply_casino_tax(
        self,
        amount: int,
        won: bool,
    ) -> tuple[int, int]:
        try:
            final_amount, tax_amount = await self.calculate_tax(abs(amount), "casino")
            actual_amount = (1 if won else -1) * final_amount

            self.fed_state.update(
                {
                    "tax_collected": self.fed_state.get("tax_collected", 0)
                    + tax_amount,
                    "reserve": self.fed_state["reserve"] + tax_amount,
                }
            )

            return actual_amount, tax_amount

        except Exception as e:
            logger.error(f"Error applying casino tax: {e}", exc_info=True)
            return (-1 if not won else 1) * amount, 0

    async def check_debt_limit(self, user_id: str, potential_debt: int) -> bool:
        return min(
            0, (await self.get_user_elo(user_id)).get("points", 0)
        ) + potential_debt >= self.cfg.fed.get("max_debt", -100_000)

    async def apply_debt_interest(self, user_id: str) -> None:
        user_data = await self.get_user_elo(user_id)
        if (points := user_data.get("points", 0)) >= 0:
            return

        interest = int(abs(points) * self.cfg.fed["interest_rate"])
        user_data["points"] = points - interest

        await self.update_user_elo(user_id, user_data)
        await self.log_points_transaction(
            user_id, -interest, "Debt interest charge", "debt_interest"
        )

    async def log_points_transaction(
        self, user_id: str, amount: int, reason: str, transaction_type: str
    ) -> None:
        logs = self.elo.setdefault("logs", [])
        now = datetime.now(timezone.utc)
        logs[max(-999, -len(logs)) :] = [
            {
                "user_id": user_id,
                "amount": amount,
                "reason": reason,
                "type": transaction_type,
                "timestamp": now.isoformat(),
                "formatted_timestamp": format_discord_timestamp(now),
            }
        ]
        await self.save_elo()


# Controller


class EconELO(interactions.Extension):
    def __init__(self, bot: interactions.Client):
        self.bot = bot
        self.model = Model()
        self.users: Optional[dict[str, Any]] = None
        self.roles: Optional[dict[str, Any]] = None
        self.invite_cache: dict[str, int] = {}

        self.cfg = Config()
        self.OFFICER_ROLE_ID = self.cfg.admin_role_id
        self.LOG_CHANNEL_ID = self.cfg.log_channel_id
        self.LOG_FORUM_ID = self.cfg.log_forum_id
        self.LOG_POST_ID = self.cfg.log_post_id
        self.GUILD_ID = self.cfg.guild_id

        self.REWARD_ROLES = set(self.cfg.reward_roles)
        self.daily_reward = self.cfg.daily_points
        self.role_rewards = self.cfg.reward_amounts

        self.welcome_base_points = self.cfg.welcome_base_points
        self.newbie_tasks = self.cfg.newbie_tasks

        self.message_task = self.cfg.message_reward
        self.invite_task = self.cfg.invite_reward
        self.response_task = self.cfg.reaction_reward
        self.economy_control = self.cfg.economy_control

        asyncio.create_task(self.initialize_data())

    async def initialize_data(self) -> None:
        try:
            await self.model.load_elo()
            self.users = self.model.elo.setdefault("users", {})
            self.roles = self.model.elo.setdefault("roles", {})

            if self.users is not None:
                self.users.update(
                    {
                        user_id: {**self.cfg.user_default, **user_data}
                        for user_id, user_data in self.users.items()
                    }
                )

            await self.model.save_elo()

        except Exception as e:
            logger.error(f"Initialization failed: {e!r}", exc_info=True)
            raise

    # Check methods

    async def check_authorization(self, ctx: interactions.SlashContext) -> bool:
        if not {role.id for role in ctx.author.roles} & {self.OFFICER_ROLE_ID}:
            await self.send_error(ctx, "You don't have permission to use this command.")
            return False
        return True

    # View methods

    async def create_embed(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        color: Union[EmbedColor, int] = EmbedColor.INFO,
        fields: Optional[List[Dict[str, Union[str, bool]]]] = None,
        timestamp: Optional[datetime] = None,
    ) -> interactions.Embed:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        color_value: int = color.value if isinstance(color, EmbedColor) else color

        embed: interactions.Embed = interactions.Embed(
            title=title, description=description, color=color_value, timestamp=timestamp
        )

        if fields:
            for field in fields:
                embed.add_field(
                    name=field.get("name", ""),
                    value=field.get("value", ""),
                    inline=field.get("inline", True),
                )

        guild: Optional[interactions.Guild] = await self.bot.fetch_guild(self.GUILD_ID)
        if guild and guild.icon:
            embed.set_footer(text=guild.name, icon_url=guild.icon.url)
        else:
            embed.set_footer(text="鍵政大舞台")

        return embed

    @functools.lru_cache(maxsize=1)
    def get_log_channels(self) -> tuple[int, int, int]:
        return (
            self.LOG_CHANNEL_ID,
            self.LOG_POST_ID,
            self.LOG_FORUM_ID,
        )

    async def send_response(
        self,
        ctx: Optional[
            Union[
                interactions.SlashContext,
                interactions.InteractionContext,
                interactions.ComponentContext,
            ]
        ],
        title: str,
        message: str,
        color: EmbedColor,
        log_to_channel: bool = True,
        ephemeral: bool = True,
    ) -> None:
        embed: interactions.Embed = await self.create_embed(title, message, color)

        if ctx:
            await ctx.send(embed=embed, ephemeral=ephemeral)

        if log_to_channel:
            log_channel_id, log_post_id, log_forum_id = self.get_log_channels()
            # await self.send_to_channel(log_channel_id, embed)
            await self.send_to_forum_post(log_forum_id, log_post_id, embed)

    # async def send_to_channel(self, channel_id: int, embed: interactions.Embed) -> None:
    #     try:
    #         channel = await self.bot.fetch_channel(channel_id)

    #         if not isinstance(
    #             channel := (
    #                 channel if isinstance(channel, interactions.GuildText) else None
    #             ),
    #             interactions.GuildText,
    #         ):
    #             logger.error(f"Channel ID {channel_id} is not a valid text channel.")
    #             return

    #         await channel.send(embed=embed)

    #     except NotFound as nf:
    #         logger.error(f"Channel with ID {channel_id} not found: {nf!r}")
    #     except Exception as e:
    #         logger.error(
    #             f"Error sending message to channel {channel_id}: {e!r}", exc_info=True
    #         )

    async def send_to_forum_post(
        self, forum_id: int, post_id: int, embed: interactions.Embed
    ) -> None:
        try:
            if not isinstance(
                forum := await self.bot.fetch_channel(forum_id), interactions.GuildForum
            ):
                logger.error(f"Channel ID {forum_id} is not a valid forum channel.")
                return

            if not isinstance(
                thread := await forum.fetch_post(post_id),
                interactions.GuildPublicThread,
            ):
                logger.error(f"Post with ID {post_id} is not a valid thread.")
                return

            await thread.send(embed=embed)

        except NotFound:
            logger.error(f"{forum_id=}, {post_id=} - Forum or post not found")
        except Exception as e:
            logger.error(
                f"Forum post error [{forum_id=}, {post_id=}]: {e!r}", exc_info=True
            )

    async def send_error(
        self,
        ctx: Optional[
            Union[
                interactions.SlashContext,
                interactions.InteractionContext,
                interactions.ComponentContext,
            ]
        ],
        message: str,
        log_to_channel: bool = False,
        ephemeral: bool = True,
    ) -> None:
        await self.send_response(
            ctx, "Error", message, EmbedColor.ERROR, log_to_channel, ephemeral
        )

    async def send_success(
        self,
        ctx: Optional[
            Union[
                interactions.SlashContext,
                interactions.InteractionContext,
                interactions.ComponentContext,
            ]
        ],
        message: str,
        log_to_channel: bool = False,
        ephemeral: bool = True,
    ) -> None:
        await self.send_response(
            ctx, "Success", message, EmbedColor.INFO, log_to_channel, ephemeral
        )

    # Commands

    module_base = interactions.SlashCommand(
        name="econelo", description="Economy and ELO points management system"
    )

    module_group_debug = module_base.group(
        name="debug",
        description="Administrative commands for managing the points system",
    )

    module_group_claim = module_base.group(
        name="claim", description="Claim your daily, weekly and monthly point rewards"
    )

    module_group_view = module_base.group(
        name="view", description="View your points balance, history and leaderboards"
    )

    module_group_fed = module_base.group(
        name="fed",
        description="Bank operations like transfers, deposits and withdrawals",
    )

    module_group_casino = module_base.group(
        name="casino",
        description="Play games and gamble your points for a chance to win more",
    )

    module_group_help = module_base.group(
        name="help",
        description="Get help with EconELO commands",
    )

    # Command (Mint)

    @module_group_debug.subcommand(
        "mint", sub_cmd_description="Mint points for the fed (admin only)"
    )
    @interactions.slash_option(
        name="amount",
        description="Amount of points to mint",
        opt_type=interactions.OptionType.INTEGER,
        required=True,
        min_value=1,
    )
    async def debug_mint(
        self,
        ctx: interactions.SlashContext,
        amount: int,
    ) -> None:
        if not await self.check_authorization(ctx):
            return

        try:
            fed_state = self.model.fed_state
            previous_balance = fed_state["balance"]
            fed_state["balance"] += amount

            await self.model.update_market_state()
            await self.model.log_points_transaction(
                "fed", amount, f"Points minted by {ctx.author.username}", "admin_mint"
            )

            logger.info(
                f"Casino points minted by {ctx.author.id}: {amount:,} points (previous balance: {previous_balance:,})"
            )

            await self.send_success(
                ctx,
                f"Minted `{amount:,}` points for the casino. Previous balance: `{previous_balance:,}`. New balance: `{fed_state['balance']:,}`.",
                log_to_channel=True,
            )

        except Exception as e:
            logger.error(f"Error minting points: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while minting points.")

    # Command (Export)

    @module_group_debug.subcommand(
        sub_cmd_name="export",
        sub_cmd_description="Export files from the extension directory (admin only)",
    )
    @interactions.slash_option(
        name="type",
        description="Type of files to export",
        required=True,
        opt_type=interactions.OptionType.STRING,
        autocomplete=True,
        argument_name="file_type",
    )
    @interactions.slash_default_member_permission(
        interactions.Permissions.ADMINISTRATOR
    )
    async def debug_export(
        self, ctx: interactions.SlashContext, file_type: str
    ) -> None:
        if not await self.check_authorization(ctx):
            return

        await ctx.defer(ephemeral=True)
        filename: str = ""

        if not os.path.exists(BASE_DIR):
            return await self.send_error(ctx, "Extension directory does not exist.")

        if file_type != "all" and not os.path.isfile(os.path.join(BASE_DIR, file_type)):
            return await self.send_error(
                ctx, f"File `{file_type}` does not exist in the extension directory."
            )

        try:
            async with aiofiles.tempfile.NamedTemporaryFile(
                prefix="export_", suffix=".tar.gz", delete=False
            ) as afp:
                filename = afp.name
                base_name = filename[:-7]

                await aioshutil.make_archive(
                    base_name,
                    "gztar",
                    BASE_DIR,
                    "." if file_type == "all" else file_type,
                )

            if not os.path.exists(filename):
                return await self.send_error(ctx, "Failed to create archive file.")

            file_size = os.path.getsize(filename)
            if file_size > 8_388_608:
                return await self.send_error(
                    ctx, "Archive file is too large to send (>8MB)."
                )

            await ctx.send(
                (
                    "All extension files attached."
                    if file_type == "all"
                    else f"File `{file_type}` attached."
                ),
                files=[interactions.File(filename)],
            )

        except PermissionError:
            logger.error(f"Permission denied while exporting {file_type}")
            await self.send_error(ctx, "Permission denied while accessing files.")
        except Exception as e:
            logger.error(f"Error exporting {file_type}: {e}", exc_info=True)
            await self.send_error(
                ctx, f"An error occurred while exporting {file_type}: {str(e)}"
            )
        finally:
            if filename and os.path.exists(filename):
                try:
                    os.unlink(filename)
                except Exception as e:
                    logger.error(f"Error cleaning up temp file: {e}", exc_info=True)

    @debug_export.autocomplete("type")
    async def autocomplete_debug_export_type(
        self, ctx: interactions.AutocompleteContext
    ) -> None:
        choices: list[dict[str, str]] = [{"name": "All Files", "value": "all"}]

        try:
            if os.path.exists(BASE_DIR):
                files = [
                    f
                    for f in os.listdir(BASE_DIR)
                    if os.path.isfile(os.path.join(BASE_DIR, f))
                    and not f.startswith(".")
                ]

                choices.extend({"name": file, "value": file} for file in sorted(files))
        except PermissionError:
            logger.error("Permission denied while listing files")
            choices = [{"name": "Error: Permission denied", "value": "error"}]
        except Exception as e:
            logger.error(f"Error listing files: {e}", exc_info=True)
            choices = [{"name": f"Error: {str(e)}", "value": "error"}]

        await ctx.send(choices[:25])

    # Command (Adjust)

    @module_group_debug.subcommand(
        "adjust", sub_cmd_description="Adjust points for a user or role (admin only)"
    )
    @interactions.slash_option(
        name="amount",
        description="Amount of points to adjust (positive to add, negative to subtract)",
        opt_type=interactions.OptionType.INTEGER,
        required=True,
    )
    @interactions.slash_option(
        name="target",
        description="User or role to adjust points for",
        opt_type=interactions.OptionType.MENTIONABLE,
        required=True,
    )
    async def adjust_points(
        self,
        ctx: interactions.SlashContext,
        amount: int,
        target: Union[interactions.Member, interactions.Role],
    ) -> None:
        if not await self.check_authorization(ctx):
            return

        try:
            target_id: str = str(target.id)
            target_mention: str = target.mention

            if isinstance(target, interactions.Member):
                user_data: dict = await self.model.get_user_elo(target_id)
                previous_balance = user_data.get("points", 0)
                new_balance = max(0, previous_balance + amount)

                user_data |= {
                    "points": new_balance,
                    "total_points": user_data.get("total_points", 0)
                    + (amount if amount > 0 else 0),
                    "statistics": {
                        **user_data.get("statistics", {}),
                        "quality_contributions": user_data.get("statistics", {}).get(
                            "quality_contributions", 0
                        )
                        + (1 if amount > 0 else 0),
                    },
                    "market_participation": {
                        **user_data.get("market_participation", {}),
                        "reputation": max(
                            0,
                            min(
                                100,
                                user_data.get("market_participation", {}).get(
                                    "reputation", 100
                                )
                                + amount // 10,
                            ),
                        ),
                    },
                    "skills": {
                        skill: min(100, value + (amount // 20 if amount > 0 else 0))
                        for skill, value in user_data.get("skills", {}).items()
                    },
                }

                await self.model.update_user_elo(target_id, user_data)
                await self.model.log_points_transaction(
                    target_id,
                    amount,
                    f"Manual adjustment by {ctx.author.username}",
                    "admin_adjust",
                )

                if new_level := next(
                    (
                        int(level)
                        for level, data in sorted(
                            self.cfg.levels.items(), key=lambda x: int(x[0])
                        )
                        if new_balance >= data["points"]
                        and int(level) > user_data.get("level", 1)
                    ),
                    None,
                ):
                    level_data = self.cfg.levels[str(new_level)]
                    user_data["level"] = new_level
                    user_data.setdefault("titles", []).append(level_data["title"])

                    if role_id := level_data.get("role_id"):
                        try:
                            if role := ctx.guild.get_role(role_id):
                                await target.add_role(role)
                        except Exception as e:
                            logger.error(
                                f"Failed to add level role: {e}", exc_info=True
                            )

                    await self.model.update_user_elo(target_id, user_data)

            else:
                if self.roles is None:
                    self.roles = {}
                role_data = self.roles.setdefault(target_id, {"points": 0})
                previous_balance = role_data["points"]
                role_data["points"] = new_balance = max(0, previous_balance + amount)

            await self.model.save_elo()

            logger.info(
                f"Points adjusted by {ctx.author.id} for {'user' if isinstance(target, interactions.Member) else 'role'} "
                f"{target_id}: from {previous_balance} to {new_balance} (change: {amount})"
            )

            await self.send_success(
                ctx,
                f"{'Added' if amount > 0 else 'Removed'} {abs(amount)} points {'to' if amount > 0 else 'from'} {target_mention}'s balance. New balance: {new_balance}.",
                log_to_channel=True,
            )

        except Exception as e:
            logger.error(f"Error adjusting points: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while adjusting points.")

    # Command (Quality Bonus)

    @module_group_debug.subcommand(
        "quality",
        sub_cmd_description="Add quality bonus points to a message (admin only)",
    )
    @interactions.slash_option(
        name="message",
        description="Link to the message",
        opt_type=interactions.OptionType.STRING,
        required=True,
        argument_name="message_link",
    )
    @interactions.slash_option(
        name="points",
        description="Bonus points to award",
        opt_type=interactions.OptionType.INTEGER,
        required=True,
        min_value=1,
        max_value=50,
    )
    @interactions.slash_option(
        name="reason",
        description="Reason for the quality bonus",
        opt_type=interactions.OptionType.STRING,
        required=True,
    )
    async def set_quality_bonus(
        self,
        ctx: interactions.SlashContext,
        message_link: str,
        points: int,
        reason: str,
    ) -> None:
        if not await self.check_authorization(ctx):
            return

        try:
            channel_id, message_id = [int(x) for x in message_link.rsplit("/", 2)[-2:]]

            if not (channel := await self.bot.fetch_channel(channel_id)) or not (
                message := await channel.fetch_message(message_id)
            ):
                await self.send_error(ctx, "Message not found.")
                return

            user_id = str(message.author.id)
            user_elo = await self.model.get_user_elo(user_id)

            quality_multiplier = self.model.cfg.message_reward["bonuses"]["quality"][
                "multiplier"
            ]
            adjusted_points = points * quality_multiplier

            user_elo |= {
                "points": user_elo.get("points", 0) + adjusted_points,
                "total_points": user_elo.get("total_points", 0) + adjusted_points,
                "statistics": {
                    **user_elo.get("statistics", {}),
                    "quality_contributions": user_elo.get("statistics", {}).get(
                        "quality_contributions", 0
                    )
                    + 1,
                },
                "skills": {
                    **user_elo.get("skills", {}),
                    "creativity": min(
                        100,
                        user_elo.get("skills", {}).get("creativity", 0) + points // 10,
                    ),
                    "engagement": min(
                        100,
                        user_elo.get("skills", {}).get("engagement", 0) + points // 10,
                    ),
                },
            }

            tasks = [
                self.model.update_user_elo(user_id, user_elo),
                self.model.log_points_transaction(
                    user_id,
                    adjusted_points,
                    f"Quality bonus: {reason}",
                    "quality_bonus",
                ),
                self.send_success(
                    ctx,
                    f"Awarded {adjusted_points} quality bonus points to {message.author.mention} for: {reason}.",
                    log_to_channel=True,
                ),
                message.add_reaction(self.model.cfg.reaction_reward["emoji"]),
            ]

            for task in tasks[:-1]:
                await task

            try:
                await tasks[-1]
            except Exception as e:
                logger.debug(f"Failed to add reaction: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error setting quality bonus: {e}", exc_info=True)
            await self.send_error(
                ctx, "An error occurred while setting the quality bonus."
            )

    # Command (Claim Daily)

    @module_group_claim.subcommand(
        "daily", sub_cmd_description="Claim your daily reward"
    )
    async def claim_daily(self, ctx: interactions.SlashContext) -> None:
        try:
            author_id = str(ctx.author.id)
            user_elo = await self.model.get_user_elo(author_id)
            now = datetime.now(timezone.utc)

            if last_claim := user_elo.get("last_daily"):
                if (now - datetime.fromisoformat(last_claim)).total_seconds() < 86400:
                    next_claim = datetime.fromisoformat(last_claim) + timedelta(days=1)
                    await self.send_error(
                        ctx,
                        f"You can claim your daily reward again at {format_discord_timestamp(next_claim)}",
                    )
                    return

            user_roles = frozenset(role.id for role in ctx.author.roles)
            highest_role = next(
                (
                    role_id
                    for role_id in self.model.cfg.status_roles
                    if role_id in user_roles
                ),
                None,
            )

            if not highest_role:
                await self.send_error(ctx, "You don't have any reward-eligible roles.")
                return

            daily_reward = self.model.cfg.status_amounts[highest_role]["daily"]
            final_amount, tax_amount = await self.model.calculate_tax(
                int(daily_reward), "claim"
            )
            emit_result = await self.model.emit_points(
                author_id,
                int(daily_reward),
                "daily",
                "Daily reward claim",
            )
            if not emit_result:
                await self.send_error(
                    ctx,
                    "Daily rewards are currently unavailable. Please try again later.",
                )
                return

            user_elo["last_daily"] = now
            user_elo["streaks"] = {
                **user_elo.get("streaks", {}),
                "daily_login": user_elo.get("streaks", {}).get("daily_login", 0) + 1,
            }

            tax_rate = round((tax_amount / daily_reward) * 100, 2)
            await self.model.update_user_elo(author_id, user_elo)
            await self.send_success(
                ctx,
                f"You claimed your daily reward of {final_amount:,} points (Tax: {tax_amount:,} points, Rate: {tax_rate}%).",
            )

        except Exception as e:
            logger.error(f"Failed to claim daily reward: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while claiming the reward.")

    # Command (Claim Role)

    @module_group_claim.subcommand(
        "role", sub_cmd_description="Claim your role-based rewards"
    )
    @interactions.slash_option(
        name="type",
        description="Reward type to claim",
        opt_type=interactions.OptionType.STRING,
        required=True,
        choices=[
            interactions.SlashCommandChoice(name=t.capitalize(), value=t)
            for t in ("daily", "weekly", "monthly", "seasonal", "yearly")
        ],
        argument_name="claim_type",
    )
    async def claim_role(self, ctx: interactions.SlashContext, claim_type: str) -> None:
        try:
            author_id = str(ctx.author.id)
            user_roles = frozenset(role.id for role in ctx.author.roles)
            highest_role = next(
                (
                    role_id
                    for role_id in self.model.cfg.reward_roles
                    if role_id in user_roles
                ),
                None,
            )

            if not highest_role:
                await self.send_error(ctx, "You don't have any reward-eligible roles.")
                return

            reward_amount = self.model.cfg.reward_amounts[highest_role][claim_type]
            role = await ctx.guild.fetch_role(highest_role)
            role_name = getattr(role, "name", "Unknown Role")

            user_elo = await self.model.get_user_elo(author_id)
            role_status = user_elo.get("role_status", {})
            now = datetime.now(timezone.utc)

            intervals = {
                "daily": 86400,
                "weekly": 604800,
                "monthly": 2592000,
                "seasonal": 7776000,
                "yearly": 31536000,
            }

            last_claim_str = role_status.get(claim_type)
            if last_claim_str:
                last_claim = datetime.fromisoformat(last_claim_str)
                time_since_claim = (now - last_claim).total_seconds()
                interval = intervals[claim_type]
                if time_since_claim < interval:
                    remaining_time = int(interval - time_since_claim)
                    await self.send_error(
                        ctx,
                        f"You'll need to wait {timedelta(seconds=remaining_time)} until {format_discord_timestamp(last_claim + timedelta(seconds=interval))} before claiming your {claim_type} role reward again.",
                    )
                    return

            final_amount, tax_amount = await self.model.calculate_tax(
                int(reward_amount), "claim"
            )
            emit_result = await self.model.emit_points(
                author_id,
                int(reward_amount),
                "role",
                f"{claim_type.capitalize()} {role_name} role reward claim",
            )

            if not emit_result:
                await self.send_error(
                    ctx,
                    f"Role `{claim_type}` rewards are currently unavailable. Please try again later.",
                )
                return

            user_elo["role_status"] = {**role_status, claim_type: now.isoformat()}
            await self.model.update_user_elo(author_id, user_elo)

            tax_rate = round((tax_amount / reward_amount) * 100, 2)
            await self.send_success(
                ctx,
                f"You claimed your {claim_type} `{role_name}` reward of `{final_amount:,}` points (Tax: {tax_amount:,} points, Rate: {tax_rate}%).",
            )

        except Exception as e:
            logger.error(f"Failed to claim role reward: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while claiming the reward.")

    # Command (Leaderboard)

    @module_group_view.subcommand(
        "leaderboard", sub_cmd_description="View points leaderboard"
    )
    async def view_leaderboard(
        self,
        ctx: interactions.SlashContext,
    ) -> None:
        try:
            await ctx.defer()

            users = {
                k: v
                for k, v in self.model.elo.get("users", {}).items()
                if v.get("points", 0) > 0
            }

            sorted_users = sorted(
                users.items(),
                key=lambda x: (x[1].get("points", 0), -int(x[0])),
                reverse=True,
            )

            chunk_size = 10
            user_chunks = [
                sorted_users[i : i + chunk_size]
                for i in range(0, len(sorted_users), chunk_size)
            ]

            pages = []
            for chunk_idx, chunk in enumerate(user_chunks):
                users_data = [
                    (
                        i + chunk_idx * chunk_size,
                        uid,
                        data,
                        await self.bot.fetch_user(uid),
                    )
                    for i, (uid, data) in enumerate(chunk, 1)
                ]

                leaderboard_entries = [
                    f"{i}. {user.mention}: `{data.get('points', 0):,}` points"
                    for i, _, data, user in users_data
                    if user is not None
                ]

                embed = await self.create_embed(
                    title="Points Leaderboard",
                    description="\n".join(leaderboard_entries),
                )
                pages.append(embed)

            if not pages:
                await self.send_error(ctx, "No users found on the leaderboard.")
                return

            paginator = Paginator(
                client=self.bot,
                pages=pages,
                timeout_interval=60,
                show_callback_button=True,
                show_select_menu=True,
                show_back_button=True,
                show_next_button=True,
                show_first_button=True,
                show_last_button=True,
                default_title="Points Leaderboard",
                wrong_user_message="This leaderboard can only be controlled by the user who requested it.",
                hide_buttons_on_stop=True,
            )
            await paginator.send(ctx)

        except Exception as e:
            logger.error(f"Failed to display leaderboard: {e}", exc_info=True)
            await self.send_error(
                ctx, "An error occurred while fetching the leaderboard."
            )

    # Command (Profile)

    @module_group_view.subcommand("profile", sub_cmd_description="View user profile")
    @interactions.slash_option(
        name="user",
        description="User to view profile (leave empty for self)",
        opt_type=interactions.OptionType.USER,
    )
    async def view_profile(
        self,
        ctx: interactions.SlashContext,
        user: Optional[interactions.User | interactions.Member] = None,
    ) -> None:
        try:
            target_user = user or ctx.author
            user_id = str(target_user.id)

            if not (user_data := await self.model.get_user_elo(user_id)):
                await self.send_error(
                    ctx, f"No profile found for {target_user.username}"
                )
                return

            level = user_data.get("level", 1)
            level_data = self.model.cfg.levels.get(str(level), {})
            stats = user_data.get("statistics", {})
            skills = user_data.get("skills", {})

            fields = [
                {
                    "name": "Points",
                    "value": str(f"{user_data.get('points', 0):,}"),
                    "inline": True,
                },
                {
                    "name": "Total Points",
                    "value": str(f"{user_data.get('total_points', 0):,}"),
                    "inline": True,
                },
                {
                    "name": "Level",
                    "value": str(f"{level} - {level_data.get('title', 'Unknown')}"),
                    "inline": True,
                },
                {
                    "name": "Messages Sent",
                    "value": str(stats.get("messages_sent", 0)),
                    "inline": True,
                },
                {
                    "name": "Quality Contributions",
                    "value": str(stats.get("quality_contributions", 0)),
                    "inline": True,
                },
                {
                    "name": "Weekly Activity",
                    "value": str(user_data.get("weekly_activity", 0)),
                    "inline": True,
                },
            ]

            if skills:
                fields.append(
                    {
                        "name": "Skills",
                        "value": "\n".join(
                            f"- {skill.title()}: `{value}/100`"
                            for skill, value in skills.items()
                        ),
                        "inline": True,
                    }
                )

            embed = await self.create_embed(
                title=f"{target_user.mention}'s Profile",
                fields=[
                    {
                        "name": str(field["name"]),
                        "value": str(field["value"]),
                        "inline": bool(field["inline"]),
                    }
                    for field in fields
                ],
            )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Failed to display profile: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while fetching the profile.")

    # Command (Casino Flip)

    @module_group_casino.subcommand(
        sub_cmd_name="flip",
        sub_cmd_description="Bet points on coin flip",
    )
    @interactions.slash_option(
        name="bet",
        description="Amount of points to bet",
        required=True,
        opt_type=interactions.OptionType.INTEGER,
        min_value=1,
    )
    @interactions.slash_option(
        name="choice",
        description="Heads or Tails",
        required=True,
        opt_type=interactions.OptionType.STRING,
        choices=[
            interactions.SlashCommandChoice(name="Heads", value="heads"),
            interactions.SlashCommandChoice(name="Tails", value="tails"),
        ],
    )
    @interactions.slash_option(
        name="multiplier",
        description="Betting multiplier (higher risk, higher reward)",
        required=True,
        opt_type=interactions.OptionType.STRING,
        choices=[
            interactions.SlashCommandChoice(name="1.5x (Safe)", value="1.5"),
            interactions.SlashCommandChoice(name="2x (Normal)", value="2"),
            interactions.SlashCommandChoice(name="3x (Risky)", value="3"),
        ],
    )
    @interactions.slash_option(
        name="opponent",
        description="Member to play against (default: fed)",
        opt_type=interactions.OptionType.USER,
    )
    async def casino_flip(
        self,
        ctx: interactions.SlashContext,
        bet: int,
        choice: str,
        multiplier: str,
        opponent: Optional[interactions.Member] = None,
    ) -> None:
        try:
            user_id, bet_multiplier = str(ctx.author.id), float(multiplier)
            user_data = await self.model.get_user_elo(user_id)
            current_points = user_data.get("points", 0)

            if opponent:
                opponent_id = str(opponent.id)
                if opponent_id == user_id:
                    await self.send_error(ctx, "You cannot gamble with yourself!")
                    return

                opponent_data = await self.model.get_user_elo(opponent_id)
                opponent_points = opponent_data.get("points", 0)
                adjusted_bet = int(bet * bet_multiplier)

                if (
                    insufficient := next(
                        (
                            p
                            for p in (opponent_points, current_points)
                            if p < adjusted_bet
                        ),
                        None,
                    )
                ) is not None:
                    await self.send_error(
                        ctx,
                        f"{'Your opponent' if insufficient == opponent_points else 'You'} don't have enough points! Balance: {insufficient:,}",
                    )
                    return

                result = ("heads", "tails")[random.getrandbits(1)]
                won = choice == result
                base_points = int(bet * float(multiplier))
                points_delta, tax_amount = await self.model.apply_casino_tax(
                    base_points, won
                )

                new_points = current_points + points_delta
                opponent_new_points = opponent_points - points_delta

                stats_updates = (
                    (user_data.setdefault("statistics", {}), won, points_delta),
                    (
                        opponent_data.setdefault("statistics", {}),
                        not won,
                        -points_delta,
                    ),
                )

                for stats, win, pd in stats_updates:
                    stats |= {
                        k: stats.get(k, 0) + v
                        for k, v in {
                            "gambles": 1,
                            "gamble_wins": win,
                            "points_gambled": bet,
                            "points_won": max(pd, 0),
                            "points_lost": abs(min(pd, 0)),
                            "tax_paid": tax_amount,
                            "debt": min(0, new_points),
                        }.items()
                    }

                user_data["points"], opponent_data["points"] = (
                    new_points,
                    opponent_new_points,
                )
                update_tasks = [
                    self.model.update_user_elo(user_id, user_data),
                    self.model.update_user_elo(opponent_id, opponent_data),
                    self.model.log_points_transaction(
                        user_id,
                        points_delta,
                        f"P2P coin flip vs {opponent.username}: {'won' if won else 'lost'} {bet} points (x{bet_multiplier})",
                        "p2p_flip",
                    ),
                ]

            else:
                max_bet = int(
                    self.model.fed_state["balance"]
                    * self.model.cfg.fed["max_bet_ratio"]
                )
                adjusted_bet = int(bet * bet_multiplier)

                if adjusted_bet > max_bet:
                    await self.send_error(ctx, f"Maximum bet is {max_bet:,} points!")
                    return
                if adjusted_bet > current_points:
                    await self.send_error(
                        ctx,
                        f"You don't have enough points! Current balance: {current_points:,}",
                    )
                    return

                result = ("heads", "tails")[random.getrandbits(1)]
                won = choice == result
                base_points = int(bet * float(multiplier))
                points_delta, tax_amount = await self.model.apply_casino_tax(
                    base_points, won
                )
                new_points = current_points + points_delta

                fed_state = self.model.fed_state
                fed_state.update(
                    {
                        "total_bets": fed_state["total_bets"] + bet,
                        "balance": fed_state["balance"] - points_delta,
                        "total_payouts": fed_state["total_payouts"]
                        + (points_delta if won else 0),
                        "total_debt": fed_state.get("total_debt", 0)
                        + min(0, new_points),
                    }
                )

                stats = user_data.setdefault("statistics", {})
                stats |= {
                    k: stats.get(k, 0) + v
                    for k, v in {
                        "gambles": 1,
                        "gamble_wins": won,
                        "points_gambled": bet,
                        "points_won": max(points_delta, 0),
                        "points_lost": abs(min(points_delta, 0)),
                        "tax_paid": tax_amount,
                        "debt": min(0, new_points),
                    }.items()
                }
                user_data["points"] = new_points

                update_tasks = [
                    self.model.update_user_elo(user_id, user_data),
                    self.model.update_market_state(),
                    self.model.log_points_transaction(
                        user_id,
                        points_delta,
                        f"Casino coin flip: {'won' if won else 'lost'} {bet} points (x{bet_multiplier})",
                        "casino_flip",
                    ),
                ]

            result_emoji, choice_emoji = (
                "🌝" if x == "heads" else "🌚" for x in (result, choice)
            )

            tax_text = f" (Tax: {tax_amount:,})" if tax_amount else ""
            debt_text = (
                f" [DEBT: {abs(min(new_points, 0)):,}]" if new_points < 0 else ""
            )

            description = f"You {'won' if won else 'lost'}! The coin landed on {result} {result_emoji}! Multiplier: {multiplier}x. Points: {abs(points_delta):,}{tax_text}{debt_text}"

            if opponent:
                description += f"\nFed's balance: {opponent_new_points:,}"

            embed = await self.create_embed(
                title=f"Coin Flip | {choice_emoji} vs {result_emoji}",
                description=description,
                color=EmbedColor.INFO if won else EmbedColor.ERROR,
            )

            embed.add_field(
                name="Current Balance",
                value=f"{new_points:,}",
                inline=True,
            )

            await asyncio.gather(*update_tasks)
            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in fed flip: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while processing your bet.")

    # Command (Casino Dice)

    @module_group_casino.subcommand(
        sub_cmd_name="dice",
        sub_cmd_description="Roll two dice and try to beat your opponent",
    )
    @interactions.slash_option(
        name="bet",
        description="Amount of points to bet",
        required=True,
        opt_type=interactions.OptionType.INTEGER,
        min_value=1,
    )
    @interactions.slash_option(
        name="opponent",
        description="Member to play against (default: fed)",
        opt_type=interactions.OptionType.USER,
    )
    async def casino_dice(
        self,
        ctx: interactions.SlashContext,
        bet: int,
        opponent: Optional[interactions.Member] = None,
    ) -> None:
        try:
            user_id = str(ctx.author.id)
            user_data = await self.model.get_user_elo(user_id)
            current_points = user_data.get("points", 0)

            if opponent:
                opponent_id = str(opponent.id)
                if opponent_id == user_id:
                    await self.send_error(ctx, "You cannot gamble with yourself!")
                    return

                opponent_data = await self.model.get_user_elo(opponent_id)
                opponent_points = opponent_data.get("points", 0)

                if (
                    insufficient := next(
                        (p for p in (opponent_points, current_points) if p < bet), None
                    )
                ) is not None:
                    await self.send_error(
                        ctx,
                        f"{'Your opponent' if opponent_points < bet else 'You'} don't have enough points! Balance: {insufficient:,}",
                    )
                    return

                player_dice = tuple(random.randint(1, 6) for _ in range(2))
                opponent_dice = tuple(random.randint(1, 6) for _ in range(2))

                player_total, opponent_total = sum(player_dice), sum(opponent_dice)

                def get_multiplier(dice):
                    return 3.0 if dice == (6, 6) else 2.0 if dice[0] == dice[1] else 1.0

                player_multiplier = get_multiplier(player_dice)
                opponent_multiplier = get_multiplier(opponent_dice)

                player_final = player_total * player_multiplier
                opponent_final = opponent_total * opponent_multiplier

                won = player_final > opponent_final
                base_points = int(
                    bet * (player_multiplier if won else opponent_multiplier)
                )
                points_delta, tax_amount = await self.model.apply_casino_tax(
                    base_points, won
                )

                new_points = current_points + points_delta
                opponent_new_points = opponent_points - points_delta

                stats_updates = (
                    (
                        user_data.setdefault("statistics", {}),
                        won,
                        points_delta,
                    ),
                    (
                        opponent_data.setdefault("statistics", {}),
                        not won,
                        -points_delta,
                    ),
                )

                for stats, win, pd in stats_updates:
                    stats |= {
                        k: stats.get(k, 0) + v
                        for k, v in {
                            "gambles": 1,
                            "gamble_wins": win,
                            "points_gambled": bet,
                            "points_won": max(pd, 0),
                            "points_lost": max(-pd, 0),
                            "tax_paid": tax_amount,
                            "debt": min(0, new_points),
                        }.items()
                    }

                user_data["points"], opponent_data["points"] = (
                    new_points,
                    opponent_new_points,
                )

                update_tasks = [
                    self.model.update_user_elo(user_id, user_data),
                    self.model.update_user_elo(opponent_id, opponent_data),
                    self.model.log_points_transaction(
                        user_id,
                        points_delta,
                        f"P2P dice vs {opponent.username}: {'won' if won else 'lost'} {bet} points (x{player_multiplier if won else opponent_multiplier})",
                        "p2p_dice",
                    ),
                ]

                casino_dice = None

            else:
                max_bet = int(
                    self.model.fed_state["balance"]
                    * self.model.cfg.fed["max_bet_ratio"]
                )
                if bet > max_bet:
                    await self.send_error(ctx, f"Maximum bet is {max_bet:,} points!")
                    return

                if bet > current_points:
                    await self.send_error(
                        ctx,
                        f"You don't have enough points! Current balance: {current_points:,}",
                    )
                    return

                player_dice = tuple(random.randint(1, 6) for _ in range(2))
                casino_dice = tuple(random.randint(1, 6) for _ in range(2))

                player_total, casino_total = sum(player_dice), sum(casino_dice)

                def get_multiplier(dice):
                    return 3.0 if dice == (6, 6) else 2.0 if dice[0] == dice[1] else 1.0

                player_multiplier = get_multiplier(player_dice)
                casino_multiplier = get_multiplier(casino_dice)

                player_final = player_total * player_multiplier
                casino_final = casino_total * casino_multiplier

                won = player_final > casino_final
                base_points = int(
                    bet * (player_multiplier if won else casino_multiplier)
                )
                points_delta, tax_amount = await self.model.apply_casino_tax(
                    base_points, won
                )

                new_points = current_points + points_delta

                fed_state = self.model.fed_state
                fed_state.update(
                    {
                        "total_bets": fed_state["total_bets"] + bet,
                        "balance": fed_state["balance"] - points_delta,
                        "total_payouts": fed_state["total_payouts"]
                        + (points_delta if won else 0),
                    }
                )

                stats = user_data.setdefault("statistics", {})
                stats |= {
                    k: stats.get(k, 0) + v
                    for k, v in {
                        "gambles": 1,
                        "gamble_wins": won,
                        "points_gambled": bet,
                        "points_won": max(points_delta, 0),
                        "points_lost": abs(min(points_delta, 0)),
                        "tax_paid": tax_amount,
                        "debt": min(0, new_points),
                    }.items()
                }

                user_data["points"] = new_points
                update_tasks = [
                    self.model.update_user_elo(user_id, user_data),
                    self.model.update_market_state(),
                    self.model.log_points_transaction(
                        user_id,
                        points_delta,
                        f"Casino dice: {'won' if won else 'lost'} {bet} points (x{player_multiplier if won else casino_multiplier})",
                        "casino_dice",
                    ),
                ]

            player_roll = f"🎲 {player_dice[0]} + 🎲 {player_dice[1]} = {player_total}"
            opponent_roll = f"🎲 {(casino_dice or opponent_dice)[0]} + 🎲 {(casino_dice or opponent_dice)[1]} = {casino_total if casino_dice else opponent_total}"

            tax_text = f" (Tax: {tax_amount:,})" if tax_amount else ""
            debt_text = (
                f" [DEBT: {abs(min(new_points, 0)):,}]" if new_points < 0 else ""
            )

            description = f"You {'won' if won else 'lost'}! Your roll: {player_roll} (x{player_multiplier}) {'>' if won else '<'} Opponent: {opponent_roll} (x{casino_multiplier if casino_dice else opponent_multiplier}). You {'gained' if won else 'lost'} `{abs(points_delta):,}` points{tax_text}{debt_text}!"

            if opponent:
                description += f" Fed's new balance: `{opponent_new_points:,}`"

            embed = await self.create_embed(
                title="🎲 Dice Roll 🎲",
                description=description,
                color=EmbedColor.INFO if won else EmbedColor.ERROR,
            )

            embed.add_field(
                name="Current Balance",
                value=f"{new_points:,}",
                inline=True,
            )

            await asyncio.gather(*update_tasks)
            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in fed dice: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while processing your bet.")

    # Command (Casino Guess)

    @module_group_casino.subcommand(
        sub_cmd_name="guess",
        sub_cmd_description="Guess a number between 1-100 within 5 rounds",
    )
    @interactions.slash_option(
        name="bet",
        description="Amount of points to bet",
        required=True,
        opt_type=interactions.OptionType.INTEGER,
        min_value=1,
    )
    async def casino_guess(self, ctx: interactions.SlashContext, bet: int) -> None:
        try:
            user_id = str(ctx.author.id)
            user_data = await self.model.get_user_elo(user_id)
            current_points = user_data.get("points", 0)

            max_bet = int(
                self.model.fed_state["balance"] * self.model.cfg.fed["max_bet_ratio"]
            )
            if bet > max_bet or bet > current_points:
                await self.send_error(
                    ctx,
                    f"{'Maximum bet is ' + f'{max_bet:,}' if bet > max_bet else 'You do not have enough points! Current balance: ' + f'{current_points:,}'} points!",
                )
                return

            target = random.randint(1, 100)
            rounds_left = 5
            guessed_numbers = []

            def get_multiplier_text(i):
                multiplier = 10 // (2 ** (i - 1)) if i < 4 else 1 if i == 4 else 0.5
                return f"{i} round{'s' if i > 1 else ''}: x{multiplier}."

            instruction_msg = f"Guess a number between 1-100. You have 5 rounds. Fewer rounds = Higher multiplier! {' '.join(get_multiplier_text(i) for i in range(1,6))}"

            await ctx.send(
                embed=await self.create_embed(
                    title="Number Guessing Game",
                    description=instruction_msg,
                )
            )

            def check(m):
                return (
                    m.author.id == ctx.author.id
                    and m.channel_id == ctx.channel_id
                    and m.content.isdigit()
                    and 1 <= int(m.content) <= 100
                )

            while rounds_left:
                try:
                    message = await self.bot.wait_for(
                        "message_create", checks=check, timeout=30.0
                    )
                    guess = int(message.content)
                    guessed_numbers.append(guess)
                    rounds_left -= 1

                    if guess == target:
                        multiplier = {1: 10.0, 2: 5.0, 3: 2.0, 4: 1.0, 5: 0.5}[
                            5 - rounds_left
                        ]
                        base_points = int(bet * multiplier)
                        points_delta, tax_amount = await self.model.apply_casino_tax(
                            base_points, True
                        )
                        new_points = current_points + points_delta

                        fed_state = self.model.fed_state
                        fed_state |= {
                            "total_bets": fed_state["total_bets"] + bet,
                            "balance": fed_state["balance"] - points_delta,
                            "total_payouts": fed_state["total_payouts"] + points_delta,
                        }

                        user_data.setdefault("statistics", {}).update(
                            {
                                k: user_data["statistics"].get(k, 0) + v
                                for k, v in {
                                    "gambles": 1,
                                    "gamble_wins": 1,
                                    "points_gambled": bet,
                                    "points_won": points_delta,
                                    "tax_paid": tax_amount,
                                }.items()
                            }
                        )
                        user_data["points"] = new_points

                        await self.model.update_user_elo(user_id, user_data)
                        await self.model.update_market_state()
                        await self.model.log_points_transaction(
                            user_id,
                            points_delta,
                            f"Number guessing game: won {points_delta} points (x{multiplier})",
                            "casino_guess",
                        )

                        tax_text = f" (Tax: {tax_amount:,})" if tax_amount else ""
                        win_msg = f"Congratulations! You guessed the number in {5 - rounds_left} rounds! Target was: {target}. Multiplier: x{multiplier}. You won: {points_delta:,} points{tax_text}!"

                        await ctx.channel.send(
                            embed=await self.create_embed(
                                title="Number Guessing Game",
                                description=win_msg,
                            )
                        )
                        return

                    if rounds_left:
                        hint_msg = f"{'Higher' if guess < target else 'Lower'}! Rounds left: {rounds_left}. Guessed numbers: {', '.join(map(str, guessed_numbers))}."
                        await ctx.channel.send(
                            embed=await self.create_embed(
                                title="Number Guessing Game",
                                description=hint_msg,
                            )
                        )

                except asyncio.TimeoutError:
                    timeout_time = datetime.now(timezone.utc) + timedelta(seconds=30)
                    await self.send_error(
                        ctx,
                        f"Time's up! Game over. Try again after {format_discord_timestamp(timeout_time)}.",
                    )
                    return

            points_delta, tax_amount = await self.model.apply_casino_tax(bet, False)
            new_points = current_points + points_delta
            fed_state = self.model.fed_state
            fed_state |= {
                "total_bets": fed_state["total_bets"] + bet,
                "balance": fed_state["balance"] - points_delta,
            }

            user_data.setdefault("statistics", {}).update(
                {
                    k: user_data["statistics"].get(k, 0) + v
                    for k, v in {
                        "gambles": 1,
                        "points_gambled": bet,
                        "points_lost": abs(points_delta),
                        "tax_paid": tax_amount,
                    }.items()
                }
            )
            user_data["points"] = new_points

            await self.model.update_user_elo(user_id, user_data)
            await self.model.update_market_state()
            await self.model.log_points_transaction(
                user_id, points_delta, "Number guessing game: lost", "casino_guess"
            )

            tax_text = f" (Tax: {tax_amount:,})" if tax_amount else ""
            lose_msg = f"Game Over! The number was {target}. You lost {abs(points_delta):,} points{tax_text}! Your guesses: {', '.join(map(str, guessed_numbers))}."

            await ctx.channel.send(
                embed=await self.create_embed(
                    title="Number Guessing Game",
                    description=lose_msg,
                    color=EmbedColor.ERROR,
                )
            )

        except Exception as e:
            logger.error(f"Error in fed guess: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while processing your game.")

    # Command (Casino RPS)

    @module_group_casino.subcommand(
        sub_cmd_name="rps",
        sub_cmd_description="Play Rock Paper Scissors to win your bet",
    )
    @interactions.slash_option(
        name="bet",
        description="Amount of points to bet",
        required=True,
        opt_type=interactions.OptionType.INTEGER,
        min_value=1,
    )
    @interactions.slash_option(
        name="choice",
        description="Rock, Paper, or Scissors",
        required=True,
        opt_type=interactions.OptionType.STRING,
        choices=[
            interactions.SlashCommandChoice(name="✊ Rock", value="rock"),
            interactions.SlashCommandChoice(name="✋ Paper", value="paper"),
            interactions.SlashCommandChoice(name="✌️ Scissors", value="scissors"),
        ],
    )
    @interactions.slash_option(
        name="opponent",
        description="Member to play against (default: fed)",
        opt_type=interactions.OptionType.USER,
    )
    async def casino_rps(
        self,
        ctx: interactions.SlashContext,
        bet: int,
        choice: str,
        opponent: Optional[interactions.Member] = None,
    ) -> None:
        try:
            user_id = str(ctx.author.id)
            user_data = await self.model.get_user_elo(user_id)
            current_points = user_data.get("points", 0)

            choices = {
                "rock": ("scissors", "✊"),
                "paper": ("rock", "✋"),
                "scissors": ("paper", "✌️"),
            }

            if opponent:
                opponent_id = str(opponent.id)
                if opponent_id == user_id:
                    await self.send_error(ctx, "You cannot play against yourself!")
                    return

                opponent_data = await self.model.get_user_elo(opponent_id)
                opponent_points = opponent_data.get("points", 0)

                if any(bet > points for points in (current_points, opponent_points)):
                    await self.send_error(
                        ctx,
                        f"{'You' if bet > current_points else 'Your opponent'} don't have enough points! Balance: {min(current_points, opponent_points):,}",
                    )
                    return

                buttons = [
                    interactions.Button(
                        style=interactions.ButtonStyle.PRIMARY,
                        label=f"{emoji} {name.title()}",
                        custom_id=f"rps_{name}",
                    )
                    for name, (_, emoji) in choices.items()
                ]

                challenge_msg = f"{opponent.mention}: {ctx.author.mention} challenges you to Rock Paper Scissors! Bet: {bet:,} points."

                action_row = interactions.ActionRow(*buttons)
                await ctx.send(
                    embed=await self.create_embed(
                        title="Rock Paper Scissors Challenge",
                        description=challenge_msg,
                    ),
                    components=[action_row],
                )

                try:
                    component_ctx = await self.bot.wait_for_component(
                        components=action_row,
                        check=lambda c: str(c.author.id) == opponent_id,
                        timeout=30.0,
                    )
                    opponent_choice = component_ctx.ctx.custom_id.split("_")[1]
                except asyncio.TimeoutError:
                    timeout_time = datetime.now(timezone.utc) + timedelta(seconds=30)
                    await self.send_error(
                        ctx,
                        f"Challenge timed out! Try again {format_discord_timestamp(timeout_time)}.",
                    )
                    return

            else:
                max_bet = int(
                    self.model.fed_state["balance"]
                    * self.model.cfg.fed["max_bet_ratio"]
                )
                if bet > max_bet:
                    await self.send_error(ctx, f"Maximum bet is {max_bet:,} points!")
                    return
                if bet > current_points:
                    await self.send_error(
                        ctx,
                        f"You don't have enough points! Current balance: {current_points:,}",
                    )
                    return

                opponent_choice = random.choice(tuple(choices))
                fed_state = self.model.fed_state

            base_points = (
                lambda x, y: 0 if x == y else bet if y == choices[x][0] else -bet
            )(choice, opponent_choice)
            won = base_points > 0
            points_delta, tax_amount = await self.model.apply_casino_tax(
                base_points, won
            )
            new_points = current_points + points_delta

            if opponent:
                opponent_new_points = opponent_points - points_delta
                if points_delta:
                    for data, won, pd in (
                        (user_data, points_delta > 0, points_delta),
                        (opponent_data, points_delta < 0, -points_delta),
                    ):
                        data.setdefault("statistics", {}).update(
                            {
                                k: data["statistics"].get(k, 0) + v
                                for k, v in {
                                    "gambles": 1,
                                    "gamble_wins": won,
                                    "points_gambled": bet,
                                    "points_won": max(pd, 0),
                                    "points_lost": max(-pd, 0),
                                    "tax_paid": tax_amount,
                                }.items()
                            }
                        )
                user_data["points"], opponent_data["points"] = (
                    new_points,
                    opponent_new_points,
                )
                update_tasks = [
                    self.model.update_user_elo(user_id, user_data),
                    self.model.update_user_elo(opponent_id, opponent_data),
                ]
            else:
                if points_delta:
                    fed_state.update(
                        {
                            "total_bets": fed_state["total_bets"] + bet,
                            "balance": fed_state["reserve"] - points_delta,
                            "total_payouts": fed_state["total_payouts"]
                            + (points_delta if points_delta > 0 else 0),
                            "daily_emissions": {
                                **fed_state["daily_emissions"],
                                "casino_payouts": fed_state["daily_emissions"][
                                    "casino_payouts"
                                ]
                                + (points_delta if points_delta > 0 else 0),
                            },
                        }
                    )
                    user_data.setdefault("statistics", {}).update(
                        {
                            k: user_data["statistics"].get(k, 0) + v
                            for k, v in {
                                "gambles": 1,
                                "gamble_wins": points_delta > 0,
                                "points_gambled": bet,
                                "points_won": max(points_delta, 0),
                                "points_lost": max(-points_delta, 0),
                                "tax_paid": tax_amount,
                            }.items()
                        }
                    )
                user_data["points"] = new_points
                update_tasks = [
                    self.model.update_user_elo(user_id, user_data),
                    self.model.update_market_state(),
                ]

            if points_delta:
                update_tasks.append(
                    self.model.log_points_transaction(
                        user_id,
                        points_delta,
                        f"{'P2P' if opponent else 'Casino'} RPS: {'won' if points_delta > 0 else 'lost'} {bet} points",
                        f"{'p2p' if opponent else 'fed'}_rps",
                    )
                )

            tax_text = f" (Tax: {tax_amount:,})" if tax_amount else ""

            if points_delta == 0:
                description = f"It's a tie! Both chose {choices[choice][1]}"
            elif points_delta > 0:
                description = f"You won! {choices[choice][1]} beats {choices[opponent_choice][1]}! You gained {points_delta:,} points{tax_text}!"
            else:
                description = f"You lost! {choices[opponent_choice][1]} beats {choices[choice][1]}! You lost {abs(points_delta):,} points{tax_text}!"

            if opponent:
                description += (
                    f" {opponent.mention}'s new balance: {opponent_new_points:,}"
                )

            embed = await self.create_embed(
                title="Rock Paper Scissors Results",
                description=description,
                color=EmbedColor.INFO if points_delta >= 0 else EmbedColor.ERROR,
            )
            embed.add_field(
                name="Current Balance",
                value=f"{new_points:,}",
                inline=True,
            )

            for task in update_tasks:
                await task
            if opponent:
                await component_ctx.ctx.message.edit(embed=embed, components=[])
            else:
                await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in fed RPS: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while processing your game.")

    # Listener (Welcome)

    @interactions.listen(interactions.events.MemberAdd)
    async def on_member_join(self, event: interactions.events.MemberAdd) -> None:
        try:
            user_id = str(event.member.id)

            user_elo = await self.model.get_user_elo(user_id)

            user_elo |= {
                "points": user_elo.get("points", 0)
                + self.model.cfg.welcome_base_points,
                "newbie_tasks": dict.fromkeys(self.model.cfg.newbie_tasks, False),
                "join_date": datetime.now(timezone.utc).isoformat(),
            }

            await self.model.update_user_elo(user_id, user_elo)

        except Exception as e:
            logger.error(f"Error in welcome process: {str(e)}", exc_info=True)

    # Listeners (Extension)

    @interactions.listen(interactions.events.ExtensionLoad)
    async def on_extension_load(self) -> None:
        [
            task.start()
            for task in (
                self.reset_daily_invites,
                self.process_pending_rewards,
                self.reset_daily_reactions,
                self.weekly_points_distribution,
            )
        ]

    @interactions.listen(interactions.events.ExtensionUnload)
    async def on_extension_unload(self) -> None:
        for task in (
            self.reset_daily_invites,
            self.process_pending_rewards,
            self.reset_daily_reactions,
            self.weekly_points_distribution,
        ):
            task.stop()

        pending_tasks = tuple(
            filter(lambda t: t.get_name().startswith("Task-"), asyncio.all_tasks())
        )

        done_tasks: set[asyncio.Task[Any]] = set()
        if pending_tasks:
            done_tasks, _ = await asyncio.wait(
                pending_tasks,
                timeout=10.0,
                return_when=asyncio.ALL_COMPLETED,
            )

        for task in done_tasks:
            if not task.done():
                task.cancel()

    # Task (Invite Reward)

    @interactions.Task.create(interactions.IntervalTrigger(days=1))
    async def reset_daily_invites(self) -> None:
        try:
            if not (invite_task := self.invite_task):
                return
            invite_task.update({"daily_invites": {}})
            await self.model.save_elo()
            logger.info("Daily invite counts reset")
        except AttributeError:
            return
        except Exception as e:
            logger.error(f"Failed to reset daily invites: {e}", exc_info=True)
            raise

    @interactions.Task.create(interactions.IntervalTrigger(days=1))
    async def process_pending_rewards(self) -> None:
        if not all(map(self.invite_task.get, ("points", "stay_days"))):
            return

        try:
            now = datetime.now(timezone.utc)
            pending_rewards = self.invite_task.get("pending_rewards", {})

            if not isinstance(pending_rewards, dict):
                return

            to_delete: set[str] = set()
            to_reward: dict[str, int] = {}
            stay_days = self.invite_task.get("stay_days", 0)

            for inviter_id, inviter_data in (
                (k, v) for k, v in pending_rewards.items() if isinstance(v, dict)
            ):
                member_deletions = {
                    member_id
                    for member_id, data in inviter_data.items()
                    if isinstance(data, dict)
                    and (
                        not (joined_at := data.get("joined_at"))
                        or not isinstance(stay_days, (int, str))
                        or (now - datetime.fromisoformat(joined_at)).days
                        >= int(stay_days)
                    )
                }

                for member_id in member_deletions:
                    if isinstance(inviter_data[member_id], dict):
                        reward_points = inviter_data[member_id].get("points", 0)
                        if isinstance(reward_points, int):
                            to_reward[inviter_id] = (
                                to_reward.get(inviter_id, 0) + reward_points
                            )

                if member_deletions:
                    inviter_data = {
                        k: v
                        for k, v in inviter_data.items()
                        if k not in member_deletions
                    }
                    pending_rewards[inviter_id] = inviter_data

                if not inviter_data:
                    to_delete.add(inviter_id)

            for inviter_id, reward_amount in to_reward.items():
                await self.model.update_user_elo(
                    inviter_id,
                    {
                        "points": reward_amount,
                        "total_points": reward_amount,
                        "statistics": {"invites": 1},
                    },
                )

            for inviter_id in to_delete:
                pending_rewards.pop(inviter_id, None)

            if isinstance(self.invite_task, dict):
                self.invite_task["pending_rewards"] = {
                    k: v for k, v in pending_rewards.items() if v
                }
                await self.model.save_elo()

            if to_reward:
                logger.info(f"Processed rewards for {len(to_reward)} inviters")

        except Exception as e:
            logger.error(f"Error processing pending rewards: {e!r}", exc_info=True)

    @interactions.listen(interactions.events.MemberAdd)
    async def on_guild_member_add(self, event: interactions.events.MemberAdd) -> None:
        if not (reward := self.invite_task.get("points")):
            return

        try:
            guild = await self.bot.fetch_guild(self.GUILD_ID)
            current_invites = {i.code: i for i in await guild.fetch_invites()}
            now = datetime.now(timezone.utc).isoformat()

            if used_invite := next(
                (
                    invite
                    for invite in current_invites.values()
                    if invite.uses > self.invite_cache.get(invite.code, 0)
                ),
                None,
            ):
                inviter_id = str(used_invite.inviter.id)
                member_id = str(event.member.id)

                daily_invites = self.invite_task.setdefault("daily_invites", {})
                if not isinstance(daily_invites, dict):
                    self.invite_task["daily_invites"] = daily_invites = {}

                if (
                    daily_limit := self.invite_task.get("daily_limit", 0)
                ) and daily_invites.get(inviter_id, 0) >= daily_limit:
                    logger.info(f"User {inviter_id} reached daily invite limit")
                    return

                daily_invites[inviter_id] = daily_invites.get(inviter_id, 0) + 1

                if self.invite_task.get("stay_days"):
                    pending_rewards = self.invite_task.setdefault("pending_rewards", {})
                    if not isinstance(pending_rewards, dict):
                        self.invite_task["pending_rewards"] = pending_rewards = {}

                    pending_rewards.setdefault(inviter_id, {})[member_id] = {
                        "joined_at": now,
                        "points": reward,
                    }
                    logger.info(f"Pending reward added for inviter {inviter_id}")
                else:
                    await self.model.update_user_elo(
                        inviter_id,
                        {
                            "points": reward,
                            "total_points": reward,
                            "statistics": {"invites": 1},
                        },
                    )
                    logger.info(f"Instant reward given to {inviter_id}")

                self.invite_cache[used_invite.code] = used_invite.uses
                await self.model.save_elo()

        except Exception as e:
            logger.error(f"Error processing invite reward: {e}", exc_info=True)

    @interactions.listen(interactions.events.MemberRemove)
    async def on_guild_member_remove(
        self, event: interactions.events.MemberRemove
    ) -> None:
        if not (stay_days := self.invite_task.get("stay_days")):
            return

        try:
            member_id = str(event.member.id)
            now = datetime.now(timezone.utc)

            if not isinstance(
                pending_rewards := self.invite_task.get("pending_rewards", {}), dict
            ):
                self.invite_task["pending_rewards"] = pending_rewards = {}

            for inviter_id, pending in tuple(
                (i, p)
                for i, p in pending_rewards.items()
                if isinstance(p, dict) and member_id in p
            ):
                if (data := pending[member_id]) and (
                    isinstance(stay_days, (int, str))
                    and (now - datetime.fromisoformat(data["joined_at"])).days
                    < int(stay_days)
                ):
                    await self.model.update_user_elo(
                        inviter_id,
                        {
                            "points": -(points := data["points"]),
                            "total_points": -points,
                            "statistics": {"invites": -1},
                        },
                    )
                    logger.info(
                        f"Deducted {points} points from {inviter_id} due to early leave"
                    )

                del pending[member_id]
                if not pending:
                    del pending_rewards[inviter_id]

                await self.model.save_elo()
                break

        except Exception as e:
            logger.error(f"Error processing member remove: {e}", exc_info=True)

    # Task (Message Reward)

    @staticmethod
    def calculate_text_entropy(text: str) -> float:
        return (
            -sum((c := text.count(ch) / len(text)) * math.log2(c) for ch in {*text})
            if text
            else 0.0
        )

    async def evaluate_message_quality(
        self, message: interactions.Message
    ) -> Tuple[int, list[str]]:
        cfg = self.model.cfg.message_reward
        points = random.randint(*(cfg["base"][k] for k in ("min", "max")))
        bonuses_cfg = cfg.get("bonuses", {})

        content = message.content
        content_metrics = (
            len(content),
            bool(message.attachments),
            bool(
                next(re.finditer(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+", content), None)
            ),
            (
                -sum(
                    (c := content.count(ch) / len(content)) * math.log2(c)
                    for ch in {*content}
                )
                if content
                else 0.0
            ),
        )

        bonus_conditions = []
        bonus_types = [
            (
                "length",
                lambda m: m[0] >= bonuses_cfg["length"].get("threshold", 100),
                "Long text",
                2,
            ),
            ("image", lambda m: m[1], "Contains image", 3),
            ("link", lambda m: m[2], "Contains link", 2),
            (
                "entropy",
                lambda m: m[3] >= bonuses_cfg["entropy"].get("threshold", 0.7),
                "High quality content",
                3,
            ),
        ]

        for bonus_type, metric_val, desc, default_points in bonus_types:
            if bonus_cfg := bonuses_cfg.get(bonus_type):
                points = bonus_cfg.get("points", default_points)
                if metric_val(content_metrics):
                    bonus_conditions.append((True, points, f"{desc} +{points}"))

        base_points = points + sum(
            points_add for cond, points_add, _ in bonus_conditions if cond
        )

        return (
            (
                int(base_points * bonuses_cfg["quality"].get("multiplier", 1.5))
                if all(cond for cond, _, _ in bonus_conditions)
                else base_points
            ),
            [msg for cond, _, msg in bonus_conditions if cond],
        )

    async def get_daily_points_limit(self, user_id: str) -> int:
        cfg = self.model.cfg.message_reward["daily_limit"]
        today = int(datetime.now(timezone.utc).timestamp()) >> 16 << 16
        seed = hash(f"{user_id}{today}") & 0xFFFFFFFF
        random.seed(seed)
        return max(
            cfg.get("min", 50),
            cfg.get("base", 100)
            + (
                random.getrandbits(16)
                & ((1 << cfg.get("variance", 50).bit_length() + 1) - 1)
            )
            - cfg.get("variance", 50),
        )

    @interactions.listen(interactions.events.MessageCreate)
    async def on_message_create(self, event: interactions.events.MessageCreate) -> None:
        if event.message.author.bot:
            return

        user_id: str = str(event.message.author.id)
        user_elo: dict = await self.model.get_user_elo(user_id)

        if tasks := user_elo.get("newbie_tasks", {}):
            cfg = self.model.cfg.newbie_tasks
            points_gained = sum(
                cfg[task]["points"]
                for task in ("first_message", "introduce")
                if bool(not tasks.get(task) and tasks.setdefault(task, True) is True)
            )
            if points_gained:
                user_elo["points"] += points_gained
                user_elo["total_points"] += points_gained
                await self.send_success(
                    None,
                    "\n".join(
                        f"<@{user_id}> completed {cfg[task]['description']}! (+{cfg[task]['points']} points)"
                        for task in ("first_message", "introduce")
                        if tasks[task]
                    ),
                    log_to_channel=True,
                )

        user_elo["statistics"]["messages_sent"] = (
            user_elo["statistics"].get("messages_sent", 0) + 1
        )
        user_elo["weekly_activity"] = user_elo.get("weekly_activity", 0) + 1

        today = datetime.now(timezone.utc).date().isoformat()
        daily_points = user_elo.setdefault("daily_points", {"date": today, "points": 0})
        if daily_points["date"] != today:
            daily_points.update({"date": today, "points": 0})

        daily_limit = await self.get_daily_points_limit(user_id)
        if daily_points["points"] >= daily_limit:
            await self.model.update_user_elo(user_id, user_elo)
            return

        points, bonuses = await self.evaluate_message_quality(event.message)
        points = min(int(points), daily_limit - daily_points["points"])

        if points > 0:
            user_elo.update(
                {
                    "points": user_elo["points"] + points,
                    "total_points": user_elo["total_points"] + points,
                    "statistics": {
                        **user_elo.get("statistics", {}),
                        "quality_contributions": user_elo.get("statistics", {}).get(
                            "quality_contributions", 0
                        )
                        + 1,
                    },
                }
            )
            daily_points["points"] += points

            skills = user_elo.setdefault("skills", {})
            points_div_20 = points // 20
            skills.update(
                {
                    "creativity": min(
                        100, int(skills.get("creativity", 0)) + points_div_20
                    ),
                    "engagement": min(
                        100, int(skills.get("engagement", 0)) + points_div_20
                    ),
                }
            )

            update_task = self.model.update_user_elo(user_id, user_elo)

            if bonuses and points > self.model.cfg.message_reward["base"]["max"]:
                try:
                    await self.send_success(
                        None,
                        f"<@{user_id}> earned {points} points. {chr(10).join(bonuses)}.",
                        log_to_channel=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to send points notification: {e}", exc_info=True
                    )

            await update_task

    # Task (Weekly Points Distribution)

    @interactions.Task.create(interactions.IntervalTrigger(weeks=1))
    async def weekly_points_distribution(self) -> None:
        try:
            users = self.model.elo.get("users")
            if not users:
                logger.error("No users found for weekly points distribution")
                return

            bonus_multiplier = self.model.cfg.economy_control["bonus_multiplier"]
            log_channel_id = self.model.cfg.log_channel_id

            for user_id, user_data in (
                (uid, ud) for uid, ud in users.items() if ud.get("weekly_activity", 0)
            ):
                weekly_activity = user_data["weekly_activity"]
                points = int(weekly_activity * bonus_multiplier)

                stats = user_data.setdefault("statistics", {})
                streaks = user_data.setdefault("streaks", {})

                user_data.update(
                    {
                        "points": user_data.get("points", 0) + points,
                        "total_points": user_data.get("total_points", 0) + points,
                        "weekly_activity": 0,
                        "statistics": {
                            **stats,
                            "events_attended": stats.get("events_attended", 0) + 1,
                        },
                        "streaks": {
                            **streaks,
                            "weekly_participation": streaks.get(
                                "weekly_participation", 0
                            )
                            + 1,
                        },
                    }
                )

                await self.model.log_points_transaction(
                    user_id, points, "Weekly activity reward", "weekly_reward"
                )

                if level_data := await self.check_level_up(user_id):
                    try:
                        user, channel = await self.bot.fetch_user(
                            user_id
                        ), await self.bot.fetch_channel(log_channel_id)
                        if all((channel, user)):
                            await self.send_success(
                                None,
                                f"{user.mention} have reached level {level_data['new_level']} and earned the title: {level_data['title']}.",
                                log_to_channel=True,
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to send level up notification: {e}", exc_info=True
                        )
                        await self.send_error(
                            channel, "Failed to send level up notification."
                        )

            await self.model.save_elo()

        except Exception as e:
            logger.error(f"Weekly points distribution failed: {e}", exc_info=True)

    async def check_level_up(self, user_id: str) -> Optional[dict]:
        if not (user_data := self.model.elo["users"].get(str(user_id))):
            return None

        current_level, total_points = user_data.get("level", 1), user_data.get(
            "total_points", 0
        )

        try:
            guild, member = await self.bot.fetch_guild(self.model.cfg.guild_id), await (
                await self.bot.fetch_guild(self.model.cfg.guild_id)
            ).fetch_member(user_id)

            levels_data = sorted(
                self.model.cfg.levels.items(), key=lambda x: -int(x[0])
            )
            for level, data in (
                item
                for item in levels_data
                if item[1]["points"] <= total_points and int(item[0]) > current_level
            ):
                old_role_id = self.model.cfg.levels[str(current_level)].get("role_id")
                new_role_id = data.get("role_id")

                try:
                    if old_role_id and (
                        old_role := await guild.fetch_role(old_role_id)
                    ):
                        await member.remove_role(old_role)
                except Exception as e:
                    logger.error(f"Failed to remove old role: {e}", exc_info=True)

                try:
                    if new_role_id and (
                        new_role := await guild.fetch_role(new_role_id)
                    ):
                        await member.add_role(new_role)
                except Exception as e:
                    logger.error(f"Failed to add new role: {e}", exc_info=True)

                user_data["level"] = int(level)
                user_data.setdefault("titles", []).append(data["title"])
                await self.model.save_elo()

                return {
                    "new_level": int(level),
                    "title": data["title"],
                    "role_id": new_role_id,
                }

        except Exception as e:
            logger.error(f"Error managing level roles: {e}", exc_info=True)

        return None

    # Task (Reaction Reward)

    @interactions.listen(interactions.events.MessageReactionAdd)
    async def on_reaction_add(
        self, event: interactions.events.MessageReactionAdd
    ) -> None:
        try:
            if (author := event.author).bot or (emoji := str(event.emoji)) not in (
                {self.model.cfg.reaction_reward["emoji"]}
                | self.model.cfg.reaction_reward["special_emojis"].keys()
            ):
                return

            message, reactor_id = event.message, str(author.id)
            if reactor_id == (target_id := str(message.author.id)):
                return

            reactions = self.model.elo.setdefault("reactions", {})
            if (reaction_key := f"{message.id}:{reactor_id}") in reactions:
                return

            points_limit = self.model.cfg.reaction_reward["daily_limit"]
            if (
                daily_points := await self.model.get_user_daily_reaction_points(
                    target_id
                )
            ) >= points_limit:
                return

            points = min(
                self.model.cfg.reaction_reward["special_emojis"].get(
                    emoji, self.model.cfg.reaction_reward["points"]
                ),
                points_limit - daily_points,
            )
            if points <= 0:
                return

            user_elo = await self.model.get_user_elo(target_id)
            user_elo["points"] = user_elo["points"] + points
            user_elo["total_points"] = user_elo["total_points"] + points
            user_elo["daily_reactions"]["points"] = (
                user_elo["daily_reactions"]["points"] + points
            )

            if (newbie_tasks := user_elo.get("newbie_tasks")) and not newbie_tasks.get(
                "add_reaction"
            ):
                newbie_tasks["add_reaction"] = True
                cfg = self.model.cfg.newbie_tasks
                task_points = cfg["add_reaction"]["points"]
                user_elo["points"] += task_points
                user_elo["total_points"] += task_points
                await self.model.update_user_elo(reactor_id, user_elo)

                await self.send_success(
                    None,
                    f"<@{reactor_id}> completed {cfg['add_reaction']['description']}! (+{task_points} points).",
                    log_to_channel=True,
                )

            reactions[reaction_key] = {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.model.update_user_elo(target_id, user_elo)

            try:
                await self.send_success(
                    None,
                    f"<@{target_id}> received {emoji} from <@{reactor_id}> and earned {points} points.",
                    log_to_channel=True,
                )
            except Exception as e:
                logger.error(
                    f"Failed to send reaction notification: {e}", exc_info=True
                )
                await self.send_error(
                    None, "Failed to send reaction notification", log_to_channel=True
                )

            await self.model.log_points_transaction(
                target_id, points, f"Reaction reward from {reactor_id}", "reaction"
            )

            logger.info(
                f"User {target_id} received {points} points from reaction by {reactor_id}"
            )

        except Exception as e:
            logger.error(f"Error processing reaction: {e}", exc_info=True)

    @interactions.Task.create(interactions.IntervalTrigger(days=1))
    async def reset_daily_reactions(self) -> None:
        try:
            now = datetime.now(timezone.utc)
            yesterday = (now - timedelta(days=1)).isoformat()
            today = now.date().isoformat()

            if reactions := self.model.elo.get("reactions"):
                self.model.elo["reactions"] = {
                    k: v
                    for k, v in reactions.items()
                    if v.get("timestamp", "") > yesterday
                }

            if users := self.model.elo.get("users"):
                default_daily = {"date": today, "points": 0}
                for user in users.values():
                    if "daily_reactions" in user:
                        user["daily_reactions"] = default_daily.copy()

            await self.model.save_elo()
            logger.info("Daily reaction records reset")

        except Exception as e:
            logger.error(f"Error resetting daily reactions: {e}", exc_info=True)

    # Command (Help)

    @module_group_help.subcommand(
        sub_cmd_name="main",
        sub_cmd_description="Get general help about EconELO system",
    )
    async def help_main(self, ctx: interactions.SlashContext) -> None:
        try:
            embed = await self.create_embed(
                title="EconELO System Help",
                description="Welcome to the EconELO economy system! Here's what you need to know:",
                color=EmbedColor.INFO,
            )

            fields = [
                {
                    "name": "Points System",
                    "value": "Earn points through daily activities, role rewards, and casino games. Points can be used for betting and future features.",
                    "inline": True,
                },
                {
                    "name": "Levels",
                    "value": "Gain levels as you accumulate points. Each level gives you new titles and perks.",
                    "inline": True,
                },
                {
                    "name": "Federal Reserve",
                    "value": "The system includes a federal reserve that manages point supply, taxes, and interest rates.",
                    "inline": True,
                },
                {
                    "name": "Available Commands",
                    "value": "- `/econelo help claim` - Learn about claiming rewards\n"
                    "- `/econelo help casino` - Learn about casino games\n"
                    "- `/econelo help view` - Learn about viewing stats",
                    "inline": True,
                },
                {
                    "name": "Need More Help?",
                    "value": "Ask a moderator or administrator for assistance.",
                    "inline": True,
                },
            ]

            for field in fields:
                embed.add_field(
                    name=field["name"], value=field["value"], inline=field["inline"]
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error displaying main help: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while displaying help.")

    @module_group_help.subcommand(
        sub_cmd_name="casino",
        sub_cmd_description="Get help about casino games",
    )
    async def help_casino(self, ctx: interactions.SlashContext) -> None:
        try:
            embed = await self.create_embed(
                title="Casino Games Help",
                description="Try your luck with various casino games! Here are the available games:",
                color=EmbedColor.INFO,
            )

            fields = [
                {
                    "name": "Dice Roll (`/econelo casino dice`)",
                    "value": "- Bet points and roll two dice\n"
                    "- Doubles give 2x multiplier\n"
                    "- Double 6's give 3x multiplier\n"
                    "- Play against house or other players",
                    "inline": True,
                },
                {
                    "name": "Coin Flip (`/econelo casino flip`)",
                    "value": "- Bet on heads or tails\n"
                    "- Choose multiplier (1.5x, 2x, 3x)\n"
                    "- Higher multiplier = higher risk/reward\n"
                    "- Challenge players or play against house",
                    "inline": True,
                },
                {
                    "name": "Number Guess (`/econelo casino guess`)",
                    "value": "- Guess number between 1-100\n"
                    "- 5 rounds to win\n"
                    "- Earlier wins = higher multiplier\n"
                    "- Round 1: 10x → Round 5: 0.5x",
                    "inline": True,
                },
                {
                    "name": "Rock Paper Scissors (`/econelo casino rps`)",
                    "value": "- Choose rock, paper, or scissors\n"
                    "- Play against house or other players\n"
                    "- Winner takes the bet amount",
                    "inline": True,
                },
                {
                    "name": "Important Notes",
                    "value": "- All bets are taxed\n"
                    "- House has maximum bet limits\n"
                    "- Don't bet more than you can afford to lose",
                    "inline": True,
                },
            ]

            for field in fields:
                embed.add_field(
                    name=field["name"], value=field["value"], inline=field["inline"]
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error displaying casino help: {e}", exc_info=True)
            await self.send_error(
                ctx, "An error occurred while displaying casino help."
            )

    @module_group_help.subcommand(
        sub_cmd_name="claim",
        sub_cmd_description="Get help about claiming rewards",
    )
    async def help_claim(self, ctx: interactions.SlashContext) -> None:
        try:
            embed = await self.create_embed(
                title="Claiming Rewards Help",
                description="Learn how to claim various rewards in the EconELO system:",
                color=EmbedColor.INFO,
            )

            fields = [
                {
                    "name": "Daily Rewards (`/econelo claim daily`)",
                    "value": "- Claim once every 24 hours\n"
                    "- Amount based on your status role\n"
                    "- Higher status = more points\n"
                    "- Builds daily login streak",
                    "inline": True,
                },
                {
                    "name": "Role Rewards (`/econelo claim role`)",
                    "value": "- Claim rewards based on your role tier\n"
                    "- Types: daily, weekly, monthly, seasonal, yearly\n"
                    "- Higher roles get larger rewards\n"
                    "- Each type has its own cooldown",
                    "inline": True,
                },
                {
                    "name": "Status Roles",
                    "value": "\n".join(
                        f"- <@&{role_id}>" for role_id in self.model.cfg.status_roles
                    ),
                    "inline": True,
                },
                {
                    "name": "Reward Roles",
                    "value": "\n".join(
                        f"- <@&{role_id}>" for role_id in self.model.cfg.reward_roles
                    ),
                    "inline": True,
                },
                {
                    "name": "Notes",
                    "value": "- All claims are taxed based on amount\n"
                    "- Remember to claim regularly\n"
                    "- Higher roles have higher tax rates",
                    "inline": True,
                },
            ]

            for field in fields:
                embed.add_field(
                    name=field["name"], value=field["value"], inline=field["inline"]
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error displaying claim help: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while displaying claim help.")

    @module_group_help.subcommand(
        sub_cmd_name="view",
        sub_cmd_description="Get help about viewing stats and information",
    )
    async def help_view(self, ctx: interactions.SlashContext) -> None:
        try:
            embed = await self.create_embed(
                title="Viewing Information Help",
                description="Learn how to view various statistics and information:",
                color=EmbedColor.INFO,
            )

            fields = [
                {
                    "name": "Profile (`/econelo view profile`)",
                    "value": "- View your current points\n"
                    "- See total earnings\n"
                    "- Check your level and titles\n"
                    "- View activity statistics\n"
                    "- Check skill levels",
                    "inline": True,
                },
                {
                    "name": "Leaderboard (`/econelo view leaderboard`)",
                    "value": "- See top point earners\n"
                    "- Navigate through pages\n"
                    "- Compare your ranking\n"
                    "- View others' achievements",
                    "inline": True,
                },
                {
                    "name": "Statistics Shown",
                    "value": "- Current points balance\n"
                    "- Total points earned\n"
                    "- Messages and reactions\n"
                    "- Gambling statistics\n"
                    "- Achievement progress",
                    "inline": True,
                },
                {
                    "name": "Tips",
                    "value": "- Check profile regularly\n"
                    "- Track your progress\n"
                    "- Monitor your gambling stats\n"
                    "- Keep an eye on debt",
                    "inline": True,
                },
            ]

            for field in fields:
                embed.add_field(
                    name=field["name"], value=field["value"], inline=field["inline"]
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error displaying view help: {e}", exc_info=True)
            await self.send_error(ctx, "An error occurred while displaying view help.")
