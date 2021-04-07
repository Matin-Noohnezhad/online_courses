<?php

$servername = "localhost";
$user = "root";
$password = "";
$dbname = "final_project";

// Create connection
$conn = new mysqli($servername, $user, $password, $dbname);
mysqli_set_charset($conn, "utf8");
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $delete = test_input($_POST["delete"]);
    $good_id = test_input($_POST["good_id"]);
    if ($delete == 1) {
        $sql = "delete from good where id=" . $good_id;
        $result = $conn->query($sql);
    } else {
        $newName = test_input($_POST["newName"]);
        $newPrice = test_input($_POST["newPrice"]);

        if ($newName != '') {
            $sql = "update good
        set name = '" . $newName . "'
        where id=" . $good_id;
            $result = $conn->query($sql);
        }
        if ($newPrice != '') {
            $sql = "update good
        set price = '" . $newPrice . "'
        where id=" . $good_id;
            $result = $conn->query($sql);
        }
    }
    $conn->close();
}

session_start();
if (isset($_SESSION['sellerId']) && !empty($_SESSION['sellerId'])) {
    $signed_in = true;
} else {
    $signed_in = false;
}

if ($signed_in) {
    header("location:./sellerUpdateGood.php");
} else {
    header("location:./adminUpdateGood.php");
}
function test_input($data)
{
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);
    return $data;
}

?>