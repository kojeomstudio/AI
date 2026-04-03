namespace MabinogiMacro.Models;

public enum ElementType
{
    COAL_VEIN = 0,
    UI_ATTACK = 1,
    UI_INVENTORY = 2,
    UI_RIDING = 3,
    UI_MINING = 4,
    UI_CRAFT = 5,
    UI_COMPASS = 6,
    IRON_VEIN = 7,
    NORMAL_VEIN = 8,
    TREE = 9,
    UI_WORKING = 10,
    UI_RIDING_OUT = 11,
    UI_FELLING = 12,
    UI_WING = 13
}

public enum InputMethod
{
    SendInput,
    PostMessage,
    SendMessage
}

public record DetectedElement(ElementType Type, int ClassId, int X1, int Y1, int X2, int Y2)
{
    public int CenterX => (X1 + X2) / 2;
    public int CenterY => (Y1 + Y2) / 2;
}
