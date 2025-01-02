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
import aiopath
import interactions
import orjson
from interactions.client.errors import NotFound
from interactions.ext.paginators import Paginator

BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
LOG_FILE: str = os.path.join(BASE_DIR, "elo.log")
ELO_FILE: str = os.path.join(BASE_DIR, "elo.json")

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
    reward_roles: tuple[int, ...] = field(default_factory=lambda: (1234567890,))
    log_channel_id: int = 1166627731916734504
    log_forum_id: int = 1159097493875871784
    log_post_id: int = 1279118293936111707
    electoral_role_id: int = 1200043628899356702
    approved_role_id: int = 1282944839679344721
    temporary_role_id: int = 1164761892015833129
    guild_id: int = 1150630510696075404

    role_initial_points: dict[str, int] = field(
        default_factory=lambda: {"electoral": 100, "approved": 50, "temporary": 20}
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
    reward_amounts: dict[str, float] = field(
        default_factory=lambda: {
            "daily": 10.0,
            "weekly": 40.0,
            "monthly": 150.0,
            "seasonal": 500.0,
            "yearly": 2000.0,
        }
    )

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
            "special_emojis": {"🏆": 10, "💎": 8, "🌟": 7},
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
            "1": {"points": 66666, "title": "赌神", "role_id": 1292379114476671048},
            "2": {"points": 88888, "title": "赌神", "role_id": 1292379114476671048},
            "3": {
                "points": 99999,
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

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Config has no attribute `{key}`")


class Model:
    def __init__(self) -> None:
        self.elo: dict[str, Any] = {}
        self.cfg = Config()

    async def load_elo(self) -> None:
        try:
            if not await aiopath.Path(ELO_FILE).exists():
                self.elo = {}
                await self.save_elo()
                return

            async with aiofiles.open(ELO_FILE, mode="rb") as f:
                self.elo = orjson.loads(await f.read())

        except (IOError, orjson.JSONDecodeError) as e:
            logger.error("Failed to load ELO data: %r", e)
            raise
        except Exception as e:
            logger.critical("Unexpected error loading ELO data: %r", e)
            raise

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

    async def update_user_elo(self, user_id: str, update_data: dict[str, Any]) -> None:
        user_id = str(user_id)
        users = self.elo.setdefault("users", {})
        user = users.setdefault(user_id, await self.get_user_elo(user_id))

        def update_nested(key: str, value: Any) -> None:
            if key not in user:
                user[key] = self.cfg.user_default[key].copy()
            user[key].update(value)

        update_map = {
            "points": lambda v: max(0, int(v)),
            "total_points": lambda v: max(0, int(v)),
            "level": lambda v: max(1, int(v)),
            "statistics": update_nested,
            "skills": update_nested,
            "market_participation": update_nested,
            "streaks": update_nested,
        }

        for key, value in update_data.items():
            if key in update_map:
                handler = update_map[key]
                if callable(handler):
                    if isinstance(handler, type(lambda: None)):
                        user[key] = handler(value)
                    else:
                        handler(key, value)
            elif key in {
                "last_daily",
                "role_daily",
                "role_weekly",
                "role_monthly",
                "role_seasonal",
                "role_yearly",
            }:
                user[key] = value
            else:
                user[key] = value

        await self.save_elo()


class ELO(interactions.Extension):
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

        self.ELECTORAL_ROLE_ID = self.cfg.electoral_role_id
        self.APPROVED_ROLE_ID = self.cfg.approved_role_id
        self.TEMPORARY_ROLE_ID = self.cfg.temporary_role_id

        self.REWARD_ROLES = set(self.cfg.reward_roles)
        self.daily_reward = self.cfg.daily_points
        self.role_rewards = self.cfg.reward_amounts
        self.role_initial_points = self.cfg.role_initial_points

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

            if self.roles is not None:
                missing_roles = set(map(str, self.REWARD_ROLES)) - self.roles.keys()
                self.roles.update(
                    {
                        role_id: {"points": self.role_initial_points}
                        for role_id in missing_roles
                    }
                )

            await self.model.save_elo()

        except Exception as e:
            logger.error(f"Initialization failed: {e!r}")
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
        fields: Optional[List[Dict[str, str]]] = None,
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
            await self.send_to_channel(log_channel_id, embed)
            await self.send_to_forum_post(log_forum_id, log_post_id, embed)

    async def send_to_channel(self, channel_id: int, embed: interactions.Embed) -> None:
        try:
            channel = await self.bot.fetch_channel(channel_id)

            if not isinstance(
                channel := (
                    channel if isinstance(channel, interactions.GuildText) else None
                ),
                interactions.GuildText,
            ):
                logger.error(f"Channel ID {channel_id} is not a valid text channel.")
                return

            await channel.send(embed=embed)

        except NotFound as nf:
            logger.error(f"Channel with ID {channel_id} not found: {nf!r}")
        except Exception as e:
            logger.error(f"Error sending message to channel {channel_id}: {e!r}")

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
            logger.error(f"Forum post error [{forum_id=}, {post_id=}]: {e!r}")

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
        name="elo", description="ELO points management system"
    )

    module_group_admin = module_base.group(
        name="debug", description="Administrative commands for managing points"
    )

    module_group_claim = module_base.group(
        name="claim", description="Commands for claiming daily/weekly/monthly points"
    )

    module_group_view = module_base.group(
        name="view", description="Commands for viewing points and statistics"
    )

    # Adjust points for a user or role

    @module_group_admin.subcommand(
        "adjust", sub_cmd_description="Adjust points for a user or role"
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
                await self.log_points_transaction(
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
                            logger.error(f"Failed to add level role: {e}")

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
                f"{'Added' if amount > 0 else 'Removed'} {abs(amount)} points {'to' if amount > 0 else 'from'} "
                f"{target_mention}'s balance. New balance: {new_balance}.",
                log_to_channel=True,
            )

        except Exception as e:
            logger.error(f"Error adjusting points: {e}")
            await self.send_error(ctx, "An error occurred while adjusting points.")

    # Set invite reward task

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
            logger.error(f"Failed to reset daily invites: {e}")
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
            logger.error(f"Error processing pending rewards: {e!r}")

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
            logger.error(f"Error processing invite reward: {e}")

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
            logger.error(f"Error processing member remove: {e}")

    # Set message-based reward task

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
                    await event.message.channel.send(
                        f"<@{user_id}> earned {points} points!\n{chr(10).join(bonuses)}",
                        delete_after=10,
                    )
                except Exception as e:
                    logger.error(f"Failed to send points notification: {e!r}")

            await update_task

    # Add quality bonus points to a message

    @module_group_admin.subcommand(
        "quality", sub_cmd_description="Add quality bonus points to a message"
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
                self.log_points_transaction(
                    user_id,
                    adjusted_points,
                    f"Quality bonus: {reason}",
                    "quality_bonus",
                ),
                self.send_success(
                    ctx,
                    f"Awarded {adjusted_points} quality bonus points to {message.author.mention} for: {reason}",
                    log_to_channel=True,
                ),
                message.add_reaction(self.model.cfg.reaction_reward["emoji"]),
            ]

            for task in tasks[:-1]:
                await task

            try:
                await tasks[-1]
            except Exception as e:
                logger.debug(f"Failed to add reaction: {e}")

        except Exception as e:
            logger.error(f"Error setting quality bonus: {e}")
            await self.send_error(
                ctx, "An error occurred while setting the quality bonus."
            )

    # Set role for level

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

                await self.log_points_transaction(
                    user_id, points, "Weekly activity reward", "weekly_reward"
                )

                if level_data := await self.check_level_up(user_id):
                    try:
                        user, channel = await self.bot.fetch_user(
                            user_id
                        ), await self.bot.fetch_channel(log_channel_id)
                        if all((channel, user)):
                            await channel.send(
                                f"Congratulations {user.mention}! You've reached level {level_data['new_level']} "
                                f"and earned the title: {level_data['title']}"
                            )
                    except Exception as e:
                        logger.error(f"Failed to send level up notification: {e}")

            await self.model.save_elo()

        except Exception as e:
            logger.error(f"Weekly points distribution failed: {e}")

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
                    logger.error(f"Failed to remove old role: {e}")

                try:
                    if new_role_id and (
                        new_role := await guild.fetch_role(new_role_id)
                    ):
                        await member.add_role(new_role)
                except Exception as e:
                    logger.error(f"Failed to add new role: {e}")

                user_data["level"] = int(level)
                user_data.setdefault("titles", []).append(data["title"])
                await self.model.save_elo()

                return {
                    "new_level": int(level),
                    "title": data["title"],
                    "role_id": new_role_id,
                }

        except Exception as e:
            logger.error(f"Error managing level roles: {e}")

        return None

    async def log_points_transaction(
        self, user_id: str, amount: int, reason: str, transaction_type: str
    ) -> None:
        logs = self.model.elo.setdefault("logs", [])
        logs[max(-999, -len(logs)) :] = [
            {
                "user_id": user_id,
                "amount": amount,
                "reason": reason,
                "type": transaction_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
        await self.model.save_elo()

    # Set reaction reward task

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
                    f"<@{reactor_id}> completed {cfg['add_reaction']['description']}! (+{task_points} points)",
                    log_to_channel=True,
                )

            reactions[reaction_key] = {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.model.update_user_elo(target_id, user_elo)

            notification = f"<@{target_id}> received {emoji} from <@{reactor_id}> and earned {points} points!"
            try:
                await message.channel.send(notification, delete_after=10)
            except Exception as e:
                logger.error(f"Failed to send reaction notification: {e}")

            await self.log_points_transaction(
                target_id, points, f"Reaction reward from {reactor_id}", "reaction"
            )

            logger.info(
                f"User {target_id} received {points} points from reaction by {reactor_id}"
            )

        except Exception as e:
            logger.error(f"Error processing reaction: {e}")

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
            logger.error(f"Error resetting daily reactions: {e}")

    # Claim your daily reward

    @module_group_claim.subcommand(
        "daily", sub_cmd_description="Claim your daily reward"
    )
    async def claim_daily(self, ctx: interactions.SlashContext) -> None:
        try:
            author_id = (
                str(ctx.author.id) if isinstance(ctx.author.id, int) else ctx.author.id
            )
            user_elo = await self.model.get_user_elo(author_id)
            now = datetime.now(timezone.utc)

            if last_claim := user_elo.get("last_daily"):
                if (time_delta := (now - last_claim).total_seconds()) < 86400:
                    remaining = 86400 - time_delta
                    await self.send_error(
                        ctx,
                        f"You can claim your daily reward again in {remaining // 3600:.0f}h {remaining % 3600 // 60:.0f}m",
                    )
                    return

            if (daily_reward := self.model.cfg.reward_amounts["daily"]) <= 0:
                await self.send_error(ctx, "Daily rewards are currently disabled.")
                return

            points = user_elo.get("points", 0)
            total_points = user_elo.get("total_points", 0)
            streaks = user_elo.get("streaks", {})

            user_elo.update(
                {
                    "points": points + daily_reward,
                    "total_points": total_points + daily_reward,
                    "last_daily": now,
                    "streaks": {
                        **streaks,
                        "daily_login": streaks.get("daily_login", 0) + 1,
                    },
                }
            )

            await self.model.update_user_elo(author_id, user_elo)
            await self.log_points_transaction(
                author_id, int(daily_reward), "Daily reward claim", "daily_reward"
            )

            await self.send_success(
                ctx,
                f"You claimed your daily reward of {daily_reward:,} points!",
                log_to_channel=True,
            )

        except Exception as e:
            logger.error(f"Failed to claim daily reward: {e}")
            await self.send_error(ctx, "An error occurred while claiming the reward.")

    # Claim your role-based rewards

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
            if not {role.id for role in ctx.author.roles} & set(
                self.model.cfg.reward_roles
            ):
                await self.send_error(
                    ctx, "You don't have the required role to claim this reward."
                )
                return

            intervals = {
                "daily": 86_400,
                "weekly": 604_800,
                "monthly": 2_592_000,
                "seasonal": 7_776_000,
                "yearly": 31_536_000,
            }

            user_elo = await self.model.get_user_elo(author_id := str(ctx.author.id))
            role_status = user_elo.get("role_status", {})

            now = datetime.now(timezone.utc)
            if last_claim := role_status.get(claim_type):
                if (delta := (now - last_claim).total_seconds()) < intervals[
                    claim_type
                ]:
                    remaining = intervals[claim_type] - delta
                    d, h, m = (
                        int(remaining // 86400),
                        int(remaining % 86400 // 3600),
                        int(remaining % 3600 // 60),
                    )

                    time_str = "".join(
                        f"{v}{u} " for v, u in ((d, "d"), (h, "h"), (m, "m")) if v
                    ).rstrip()
                    await self.send_error(
                        ctx,
                        f"You can claim your {claim_type} role reward again in {time_str}",
                    )
                    return

            if (reward_amount := self.model.cfg.reward_amounts[claim_type]) <= 0:
                await self.send_error(
                    ctx, f"Role {claim_type} rewards are currently disabled."
                )
                return

            points = user_elo.get("points", 0)
            total_points = user_elo.get("total_points", 0)
            user_elo.update(
                {
                    "points": points + reward_amount,
                    "total_points": total_points + reward_amount,
                    "role_status": {**role_status, claim_type: now},
                }
            )

            await self.model.update_user_elo(author_id, user_elo)
            await self.log_points_transaction(
                author_id,
                int(reward_amount),
                f"{claim_type.capitalize()} role reward claim",
                f"role_{claim_type}_reward",
            )

            await self.send_success(
                ctx,
                f"You claimed your {claim_type} role reward of {reward_amount:,} points!",
                log_to_channel=True,
            )

        except Exception as e:
            logger.error(f"Failed to claim role reward: {e}")
            await self.send_error(ctx, "An error occurred while claiming the reward.")

    # View points leaderboard

    @module_group_view.subcommand(
        "leaderboard", sub_cmd_description="View points leaderboard"
    )
    async def view_leaderboard(
        self,
        ctx: interactions.SlashContext,
    ) -> None:
        try:
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
                    f"{i}. {user.username}: {data.get('points', 0):,} points"
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
                timeout_interval=120,
            )
            await paginator.send(ctx)

        except Exception as e:
            logger.error(f"Failed to display leaderboard: {e}")
            await self.send_error(
                ctx, "An error occurred while fetching the leaderboard."
            )

    # View user profile

    @module_group_view.subcommand("profile", sub_cmd_description="View user profile")
    @interactions.slash_option(
        name="user",
        description="User to view profile (leave empty for self)",
        opt_type=interactions.OptionType.USER,
    )
    async def view_profile(
        self,
        ctx: interactions.SlashContext,
        user: interactions.User = None,
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

            embed = interactions.Embed(
                title=f"{target_user.username}'s Profile", color=0x00FF00
            ).add_fields(
                interactions.EmbedField(
                    name="Points", value=f"{user_data.get('points', 0):,}", inline=True
                ),
                interactions.EmbedField(
                    name="Total Points",
                    value=f"{user_data.get('total_points', 0):,}",
                    inline=True,
                ),
                interactions.EmbedField(
                    name="Level",
                    value=f"{level} - {level_data.get('title', 'Unknown')}",
                    inline=True,
                ),
                interactions.EmbedField(
                    name="Messages Sent",
                    value=str(stats.get("messages_sent", 0)),
                    inline=True,
                ),
                interactions.EmbedField(
                    name="Quality Contributions",
                    value=str(stats.get("quality_contributions", 0)),
                    inline=True,
                ),
                interactions.EmbedField(
                    name="Weekly Activity",
                    value=str(user_data.get("weekly_activity", 0)),
                    inline=True,
                ),
            )

            if skills:
                embed.add_field(
                    name="Skills",
                    value="\n".join(
                        f"{skill.title()}: {value}/100"
                        for skill, value in skills.items()
                    ),
                    inline=True,
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Failed to display profile: {e}")
            await self.send_error(ctx, "An error occurred while fetching the profile.")

    # Welcome

    @interactions.listen(interactions.events.MemberAdd)
    async def on_member_join(self, event: interactions.events.MemberAdd) -> None:
        try:
            user_id = str(event.member.id)

            user_elo, guild = await asyncio.gather(
                self.model.get_user_elo(user_id),
                self.bot.fetch_guild(self.model.cfg.guild_id),
            )

            user_elo |= {
                "points": user_elo.get("points", 0)
                + self.model.cfg.welcome_base_points,
                "newbie_tasks": dict.fromkeys(self.model.cfg.newbie_tasks, False),
                "join_date": datetime.now(timezone.utc).isoformat(),
            }

            update_task = asyncio.create_task(
                self.model.update_user_elo(user_id, user_elo)
            )

            if self.model.cfg.temporary_role_id:
                try:
                    member = await guild.fetch_member(user_id)
                    if temp_role := await guild.fetch_role(
                        self.model.cfg.temporary_role_id
                    ):
                        await member.add_role(temp_role)
                except Exception as e:
                    logger.error(f"Failed to add temporary role: {str(e)}")

            await update_task

        except Exception as e:
            logger.error(f"Error in welcome process: {str(e)}")

    # Tasks

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
